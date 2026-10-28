"""
Run the Route B chain over one promoted BOA LCOE file.

Reads ``optimal_lcoe_*.nc``, prices hydrogen, DAC CO2 and methanol per pixel for the LCOE year
2050 (the year of the paper's process data) from each pixel's LCOE and regional WACC, and writes
three files to ``<out_dir>/<run>/``: the result NetCDF, ``<stem>_assumptions_register.csv`` and
``<stem>_provenance.json``.

The WACC is the one the LCOE was priced with. The LCOE file records the hash of the cost workbook
it used (``input_data_sha256``); the run stops if the workbook of that cost set on disk has a
different hash. Each pixel's cost key is resolved with ``single_point_run.costs_for_key``, the
lookup BOA's own query uses, so national and global-average fallbacks match the LCOE exactly.

Accepted LCOE layouts: ``schema_version`` 1 (BOA 1.9 and later, ``lcoe_basis`` per MWh
delivered), and files without a schema version written by BOA 1.8.0, which have the same
variables and LCOE definition.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import boa
import numpy as np
import xarray as xr
from boa.config.paths import PathConfig, default_root
from boa.inputs.costs import process_global_baseload_simulation_costs
from boa.model.single_point_run import costs_for_key

from lcom import register
from lcom.chain import co2_cost, lcoh, lcom
from lcom.flags import CHAIN_FLAGS, STANDING_FLAGS, check_cost_workbook_flags, lcoe_file_flags, log_flags
from lcom.paper_inputs import (
    ECB_ACCESSED,
    ECB_SERIES,
    PAPER,
    USD_PER_EUR,
    RouteBParams,
    chain_inputs,
    route_b_params,
)

logger = logging.getLogger(__name__)

PRICED_YEAR = 2050
SCHEMA_VERSIONS = (1,)
LCOE_BASIS = "per_mwh_delivered"
LEGACY_BOA_VERSIONS = ("1.8.0",)
STATUS_OK = 1
# Countries left unpriced because the BOA regions cover only part of their land (flag ``domain.greenland``).
EXCLUDED_ISO3 = frozenset({"GRL"})
ROUTE = "B: PEM electrolysis, direct CO2 hydrogenation, solid sorbent DAC with air-source heat pump"

# Attributes carried from the LCOE file; all describe the LCOE run, not this one.
CARRIED_ATTRS = (
    "schema_version",
    "lcoe_basis",
    "capacity_ceiling_applied",
    "investment_horizon_years",
    "load_density_mw_km2",
    "coverage_fraction",
    "search_params_hash",
    "era5_weather_year",
    "era5_resolution_deg",
    "run",
    "input_set",
    "cost_set",
    "input_data_sha256",
    "boa_version",
    "run_created_at",
    "run_git_sha",
    "cost_key_legend",
    "status_legend",
)

# name -> (unit, description). Every variable is float32 on (year, lat, lon).
OUTPUT_VARIABLES: dict[str, tuple[str, str]] = {
    "lcoe": ("USD/MWh", "Levelised cost of electricity from the BOA run (input, copied)"),
    "wacc": ("fraction", "Cost of capital used for every stage"),
    "lcoh_usd_per_mwh": ("USD/MWh_H2", "Levelised cost of hydrogen, lower heating value"),
    "lcoh_usd_per_kg": ("USD/kg_H2", "Levelised cost of hydrogen per kg (33.33 kWh/kg LHV)"),
    "lcoh_electricity_usd_per_mwh": ("USD/MWh_H2", "LCOH: electricity"),
    "lcoh_capital_usd_per_mwh": ("USD/MWh_H2", "LCOH: electrolyser capital charge"),
    "lcoh_fom_usd_per_mwh": ("USD/MWh_H2", "LCOH: electrolyser fixed O&M"),
    "co2_cost_usd_per_t": ("USD/tCO2", "Cost of CO2 from direct air capture"),
    "co2_electricity_usd_per_t": ("USD/tCO2", "CO2: electricity (DAC and heat pump)"),
    "co2_capital_usd_per_t": ("USD/tCO2", "CO2: DAC and heat-pump capital charge"),
    "co2_fom_usd_per_t": ("USD/tCO2", "CO2: DAC and heat-pump fixed O&M"),
    "lcom_usd_per_mwh": ("USD/MWh_MeOH", "Levelised cost of methanol, lower heating value"),
    "lcom_usd_per_t": ("USD/t_MeOH", "Levelised cost of methanol per tonne (5.528 kWh/kg LHV)"),
    "lcom_hydrogen_usd_per_mwh": ("USD/MWh_MeOH", "LCOM: hydrogen feed"),
    "lcom_electricity_usd_per_mwh": ("USD/MWh_MeOH", "LCOM: synthesis-plant electricity"),
    "lcom_co2_usd_per_mwh": ("USD/MWh_MeOH", "LCOM: CO2 feed"),
    "lcom_capital_usd_per_mwh": ("USD/MWh_MeOH", "LCOM: synthesis-plant capital charge"),
    "lcom_fom_usd_per_mwh": ("USD/MWh_MeOH", "LCOM: synthesis-plant fixed O&M"),
}


class LcoeFileError(ValueError):
    """The LCOE file cannot be priced: unknown layout, missing cost set or a mismatching workbook."""


@dataclass(frozen=True)
class RunResult:
    netcdf: Path
    register: Path
    provenance: Path


def lcoe_layout(attrs: dict[str, Any], name: str) -> str:
    """The layout of a promoted LCOE file, from its attributes; raises for a layout this code does not know."""
    if "schema_version" in attrs:
        schema = int(attrs["schema_version"])
        if schema not in SCHEMA_VERSIONS:
            raise LcoeFileError(f"{name} has schema_version {schema}; this code reads {list(SCHEMA_VERSIONS)}.")
        if attrs.get("lcoe_basis") != LCOE_BASIS:
            raise LcoeFileError(f"{name} has lcoe_basis {attrs.get('lcoe_basis')!r}, not {LCOE_BASIS!r}.")
        return f"schema {schema}"
    boa_version = str(attrs.get("boa_version", ""))
    if boa_version in LEGACY_BOA_VERSIONS:
        logger.info("%s has no schema_version: legacy layout of BOA %s.", name, boa_version)
        return f"legacy BOA {boa_version}"
    raise LcoeFileError(f"{name} has no schema_version and comes from BOA {boa_version or 'unknown'}; refused.")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wacc_grid(legend: list[str], cost_key_id: np.ndarray, costs: xr.Dataset) -> np.ndarray:
    """WACC per pixel from the pixel's cost key; NaN where the pixel has no cost key."""
    per_key = np.full(len(legend), np.nan)
    for i, key in enumerate(legend):
        if key:  # costs_for_key("") would return the global average
            per_key[i] = costs_for_key(key, costs)[2]
    return per_key[cost_key_id]


def price(
    lcoe: np.ndarray, wacc: np.ndarray, valid: np.ndarray, load_factor: float, params: RouteBParams
) -> dict[str, np.ndarray]:
    """All output layers for one year; NaN where ``valid`` is False."""
    e = np.where(valid, lcoe, 1.0)  # placeholder values keep the arithmetic finite; masked below
    w = np.where(valid, wacc, 0.05)
    hydrogen = lcoh(e, w, load_factor, params.electrolysis)
    co2 = co2_cost(e, w, load_factor, params.dac, params.heat_pump)
    methanol = lcom(e, w, load_factor, hydrogen, co2, params.methanol)
    layers = {
        "lcoe": lcoe,
        "wacc": wacc,
        "lcoh_usd_per_mwh": hydrogen.total,
        "lcoh_usd_per_kg": hydrogen.usd_per_kg(),
        "lcoh_electricity_usd_per_mwh": hydrogen.electricity,
        "lcoh_capital_usd_per_mwh": hydrogen.capital,
        "lcoh_fom_usd_per_mwh": hydrogen.fom,
        "co2_cost_usd_per_t": co2.total,
        "co2_electricity_usd_per_t": co2.electricity,
        "co2_capital_usd_per_t": co2.capital,
        "co2_fom_usd_per_t": co2.fom,
        "lcom_usd_per_mwh": methanol.total,
        "lcom_usd_per_t": methanol.usd_per_tonne(),
        "lcom_hydrogen_usd_per_mwh": methanol.hydrogen,
        "lcom_electricity_usd_per_mwh": methanol.electricity,
        "lcom_co2_usd_per_mwh": methanol.co2,
        "lcom_capital_usd_per_mwh": methanol.capital,
        "lcom_fom_usd_per_mwh": methanol.fom,
    }
    return {name: np.where(valid, layer, np.nan).astype(np.float32) for name, layer in layers.items()}


def _number_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def output_stem(lcoe_path: Path, year: int, load_factor: float | None, wacc: float | None) -> str:
    stem = lcoe_path.stem.removeprefix("optimal_lcoe_")
    suffix = "".join(f"_{tag}{_number_token(v)}" for tag, v in (("lf", load_factor), ("wacc", wacc)) if v is not None)
    return f"lcom_{stem}_y{year}{suffix}"


def _netcdf_attr(value: Any) -> Any:
    """NetCDF attributes take strings and numbers only."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (str, int, float, np.integer, np.floating)):
        return value
    return str(value)


def run(
    lcoe_path: Path,
    out_dir: Path,
    data_root: Path | None = None,
    load_factor: float | None = None,
    wacc: float | None = None,
) -> RunResult:
    """
    Price Route B methanol for every pixel of ``lcoe_path`` in the LCOE year 2050.

    Args:
        lcoe_path: A promoted BOA LCOE file.
        out_dir: Output folder; results go to ``<out_dir>/<run>/``.
        data_root: BOA data root that holds the run's cost set; default: BOA's own (``BOA_DATA_ROOT``).
        load_factor: Plant load factor in (0, 1]; default: the LCOE run's ``coverage_fraction``.
        wacc: One cost of capital for every pixel instead of the regional WACC (e.g. the paper's 0.08).
    """
    with xr.open_dataset(lcoe_path) as opened:
        missing = {"lcoe", "cost_key_id", "status"} - set(opened.variables)
        if missing or "cost_key_legend" not in opened.attrs:
            raise LcoeFileError(
                f"{lcoe_path.name} is not a promoted LCOE file (missing {sorted(missing) or 'cost_key_legend'})."
            )
        attrs = dict(opened.attrs)
        layout = lcoe_layout(attrs, lcoe_path.name)
        years = [int(y) for y in opened["year"].values]
        if PRICED_YEAR not in years:
            raise LcoeFileError(f"{lcoe_path.name} has no LCOE for {PRICED_YEAR} (years {years}).")
        lcoe = opened["lcoe"].sel(year=PRICED_YEAR).values.astype(float)
        cost_key_id = opened["cost_key_id"].values.astype(int)
        status = opened["status"].values
        lat, lon = opened["lat"].values, opened["lon"].values

    coverage = float(attrs["coverage_fraction"])
    lf = coverage if load_factor is None else float(load_factor)
    if not 0 < lf <= 1:
        raise ValueError(f"load_factor must be in (0, 1], got {lf}.")
    if wacc is not None and not 0 <= wacc < 1:
        raise ValueError(f"wacc must be in [0, 1), got {wacc}.")

    cost_set = str(attrs["cost_set"])
    path_config = PathConfig.from_root(data_root or default_root(), cost_set=cost_set)
    workbook = path_config.input_data_path
    if not workbook.exists():
        raise LcoeFileError(f"cost set '{cost_set}' of {lcoe_path.name} is not on disk ({workbook}).")
    expected_sha = str(attrs.get("input_data_sha256", ""))
    workbook_sha = file_sha256(workbook)
    if expected_sha and workbook_sha != expected_sha:
        raise LcoeFileError(
            f"The cost workbook of '{cost_set}' (sha256 {workbook_sha[:12]}) is not the one {lcoe_path.name} was "
            f"built with ({expected_sha[:12]}); the regional WACC could differ from the LCOE's."
        )

    flag_results = check_cost_workbook_flags(workbook)
    file_flags = lcoe_file_flags(attrs)
    log_flags(flag_results, file_flags)

    legend = str(attrs["cost_key_legend"]).split(",")
    has_key = np.array([bool(key) for key in legend])[cost_key_id]
    excluded = np.array([key.split(":")[0] in EXCLUDED_ISO3 for key in legend])[cost_key_id]
    if wacc is None:
        costs, _ = process_global_baseload_simulation_costs(PRICED_YEAR, workbook, path_config.cost_cache_dir)
        wacc_layer = wacc_grid(legend, cost_key_id, costs)
        wacc_basis = "regional Renewables WACC of the LCOE run (Cost of capital sheet)"
    else:
        wacc_layer = np.where(has_key, wacc, np.nan)
        wacc_basis = f"fixed override {wacc} for every pixel"
    valid = (status == STATUS_OK) & np.isfinite(lcoe) & (lcoe > 0) & np.isfinite(wacc_layer) & ~excluded

    params = route_b_params()
    layers = price(lcoe, wacc_layer, valid, lf, params)
    logger.info("Priced %d: %d pixels.", PRICED_YEAR, int(valid.sum()))

    stem = output_stem(lcoe_path, PRICED_YEAR, load_factor, wacc)
    run_dir = out_dir / str(attrs.get("run", "unknown_run"))
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = RunResult(
        run_dir / f"{stem}.nc", run_dir / f"{stem}_assumptions_register.csv", run_dir / f"{stem}_provenance.json"
    )

    created_at = datetime.now(UTC).isoformat(timespec="seconds")
    load_factor_basis = "coverage_fraction of the LCOE run" if load_factor is None else "command-line override"
    data_flags = ",".join(f.key for f in (*(r.flag for r in flag_results), *STANDING_FLAGS, *CHAIN_FLAGS, *file_flags))
    carried = {k: attrs[k] for k in CARRIED_ATTRS if k in attrs}
    out_attrs = {
        **carried,
        "capacity_ceiling_applied": attrs.get("capacity_ceiling_applied", "not recorded"),
        "title": "Levelised cost of methanol (Route B) from the BOA LCOE",
        "lcoe_file": lcoe_path.name,
        "lcoe_layout": layout,
        "priced_year": PRICED_YEAR,
        "route": ROUTE,
        "process_data": PAPER,
        "usd_per_eur": USD_PER_EUR,
        "usd_per_eur_source": f"ECB {ECB_SERIES}, 2024 annual average, accessed {ECB_ACCESSED}",
        "load_factor": lf,
        "load_factor_basis": load_factor_basis,
        "wacc_basis": wacc_basis,
        "dac_productivity": params.dac.productivity,
        "data_flags": data_flags,
        "lcom_version": version("economic-complexity"),
        "pricing_boa_version": boa.__version__,
        "created_at": created_at,
    }

    data_vars = {
        name: (("year", "lat", "lon"), layers[name][np.newaxis], {"units": unit, "description": description})
        for name, (unit, description) in OUTPUT_VARIABLES.items()
    }
    data_vars["cost_key_id"] = (
        ("lat", "lon"),
        cost_key_id.astype("int16"),
        {"description": "Index into cost_key_legend"},
    )
    data_vars["status"] = (("lat", "lon"), status.astype("int8"), {"description": "Index into status_legend"})
    out = xr.Dataset(
        data_vars=data_vars,
        coords={"year": np.array([PRICED_YEAR], dtype="int32"), "lat": lat, "lon": lon},
        attrs={k: _netcdf_attr(v) for k, v in out_attrs.items()},
    )
    encoding = {name: {"zlib": True, "complevel": 4, "dtype": "float32"} for name in OUTPUT_VARIABLES}
    out.to_netcdf(paths.netcdf, encoding=encoding)

    priced_wacc = wacc_layer[valid]
    wacc_summary = {
        "pixels_priced": int(valid.sum()),
        "min": float(priced_wacc.min()) if valid.any() else float("nan"),
        "median": float(np.median(priced_wacc)) if valid.any() else float("nan"),
        "max": float(priced_wacc.max()) if valid.any() else float("nan"),
    }
    rows = register.input_rows(chain_inputs(), PRICED_YEAR)
    rows += [
        {
            "stage": "currency",
            "parameter": "usd_per_eur",
            "year": 2024,
            "value": USD_PER_EUR,
            "unit": "USD/EUR",
            "source": f"ECB data portal, series {ECB_SERIES}, accessed {ECB_ACCESSED}",
            "status": "ecb",
            "basis": "2024 annual average reference rate",
            "data_flags": "eur-usd-annual-rate",
        },
        {
            "stage": "boa",
            "parameter": "load_factor",
            "year": "all",
            "value": lf,
            "unit": "fraction",
            "source": f"{load_factor_basis}; ERA5 weather year {attrs.get('era5_weather_year', '')}",
            "status": "boa-run" if load_factor is None else "override",
            "basis": "share of hours every plant runs",
            "data_flags": "",
        },
        {
            "stage": "boa",
            "parameter": "wacc",
            "year": PRICED_YEAR,
            "value": wacc_summary["median"],
            "unit": "fraction (median over priced pixels)",
            "source": f"{wacc_basis}; cost workbook sha256 {workbook_sha[:12]}",
            "status": "boa-run" if wacc is None else "override",
            "basis": f"min {wacc_summary['min']:.4f}, max {wacc_summary['max']:.4f}, "
            f"{wacc_summary['pixels_priced']} pixels",
            "data_flags": "wacc.country" if wacc is None else "",
        },
    ]
    register.write_register(rows, paths.register)
    register.write_provenance(
        {
            "title": out_attrs["title"],
            "created_at": created_at,
            "lcom_version": out_attrs["lcom_version"],
            "pricing_boa_version": boa.__version__,
            "lcoe_file": lcoe_path.name,
            "lcoe_layout": layout,
            "lcoe_run": {k: v for k, v in carried.items() if k != "cost_key_legend"},
            "cost_workbook_sha256": workbook_sha,
            "priced_year": PRICED_YEAR,
            "route": ROUTE,
            "process_data": PAPER,
            "usd_per_eur": {"value": USD_PER_EUR, "series": ECB_SERIES, "accessed": ECB_ACCESSED},
            "load_factor": {"value": lf, "basis": load_factor_basis},
            "wacc": {"basis": wacc_basis, **wacc_summary},
            "data_flags": register.flag_records(flag_results, file_flags),
        },
        paths.provenance,
    )
    logger.info("Wrote %s, %s and %s.", paths.netcdf.name, paths.register.name, paths.provenance.name)
    return paths
