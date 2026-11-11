"""
Electricity of the chain: the BOA solution file of the priced year, the split of each pixel's LCOE into solar,
wind and battery, and the price and load factor of the plants.

The promoted LCOE file has the LCOE only. BOA writes the cost of each technology and the served fraction to its
per-year solution file ``optimal_sol_*_GLOBAL_<year>.nc``. With ``inst`` the year-0 installation cost of a
technology, ``opex`` its fixed O&M share of CAPEX per year, ``S = sum_{t=1..H} (1 + w)^-t`` over the investment
horizon ``H``, ``load`` the baseload in MW and ``served`` the share of the baseload energy that the system
delivers:

    LCOE = sum_tech inst_tech * (1 + opex_tech * S) / (load * 8760 * S * served)

This is BOA's closed form when no technology is installed again within ``H`` (every BOA lifetime is at least
``H``). ``lcoe_split`` makes sure that the three parts add up to the LCOE of the file.

BOA supplies firm power: in the ``coverage`` share of the hours it meets the whole load; in the other hours it
delivers part of the load. The plants operate in one of two ways:

- ``full-load`` (default): the plants run at full load only in the fully met hours. The energy of the other
  hours is paid for but not used. Load factor ``coverage``, price ``LCOE * served / coverage``.
- ``part-load``: the plants also use the energy of the partly met hours, with no minimum load and no loss of
  efficiency. Load factor ``served``, price ``LCOE``.

A fixed load factor ``CF`` (an override) has the price ``LCOE * served / CF`` and must not be more than
``served``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from lcom.constants import HOURS_IN_YEAR

TECHS = ("solar", "wind", "battery")
FULL_LOAD = "full-load"
PART_LOAD = "part-load"
OPERATIONS = (FULL_LOAD, PART_LOAD)
SOLUTION_GLOB = "optimal_sol_*_GLOBAL_{year}.nc"
SOLUTION_VARIABLES = (
    "lcoe",
    "coverage",
    "served_fraction",
    "load_mw",
    "cost_of_capital",
    *(f"installation_cost_{tech}" for tech in TECHS),
)
# Attributes the solution file and the promoted LCOE file must share.
SHARED_ATTRS = ("coverage_fraction", "search_params_hash", "era5_weather_year", "investment_horizon_years")
SPLIT_RTOL = 1e-4


class SolutionFileError(ValueError):
    """The BOA solution file is missing or does not belong to the LCOE file."""


@dataclass(frozen=True)
class Solution:
    """The layers of one BOA solution file, on the grid of the LCOE file."""

    path: Path
    lcoe: np.ndarray
    coverage: np.ndarray
    served_fraction: np.ndarray
    load_mw: np.ndarray
    cost_of_capital: np.ndarray
    installation_cost: dict[str, np.ndarray]
    horizon: int


def find_solution_file(folder: Path, year: int) -> Path:
    found = sorted(folder.glob(SOLUTION_GLOB.format(year=year)))
    if len(found) != 1:
        what = "No" if not found else f"{len(found)}"
        raise SolutionFileError(
            f"{what} BOA solution file {SOLUTION_GLOB.format(year=year)} in {folder}: copy the file from "
            f"<BOA root>/runs/<run>/outputs/<weather>/<density>/<coverage>/nc/GLOBAL/, or give its path."
        )
    return found[0]


def _same(a: Any, b: Any) -> bool:
    try:
        return bool(np.isclose(float(a), float(b)))
    except (TypeError, ValueError):
        return str(a) == str(b)


def read_solution(
    path: Path, lcoe_attrs: dict[str, Any], lcoe: np.ndarray, lat: np.ndarray, lon: np.ndarray, year: int
) -> Solution:
    """
    Read the solution file and make sure it belongs to the LCOE file: same run attributes, investment year,
    grid and, where either has a value, the same LCOE.
    """
    with xr.open_dataset(path) as ds:
        missing = [v for v in SOLUTION_VARIABLES if v not in ds.variables]
        if missing:
            raise SolutionFileError(f"{path.name} is not a BOA solution file (missing {', '.join(missing)}).")
        attrs = dict(ds.attrs)
        if int(attrs.get("investment_year", -1)) != year:
            raise SolutionFileError(f"{path.name} is for investment year {attrs.get('investment_year')}, not {year}.")
        for name in SHARED_ATTRS:
            if name in lcoe_attrs and not _same(attrs.get(name), lcoe_attrs[name]):
                raise SolutionFileError(
                    f"{path.name} has {name} {attrs.get(name)!r}; the LCOE file has {lcoe_attrs[name]!r}."
                )
        if ds["lat"].shape != lat.shape or ds["lon"].shape != lon.shape or not (
            np.allclose(ds["lat"].values, lat) and np.allclose(ds["lon"].values, lon)
        ):
            raise SolutionFileError(f"{path.name} is not on the grid of the LCOE file.")
        layer = {name: ds[name].values.astype(float) for name in SOLUTION_VARIABLES}
    both = np.isfinite(layer["lcoe"]) & np.isfinite(lcoe)
    if not np.array_equal(np.isfinite(layer["lcoe"]), np.isfinite(lcoe)) or not np.allclose(
        layer["lcoe"][both], lcoe[both], rtol=1e-6
    ):
        raise SolutionFileError(f"The LCOE in {path.name} is not the LCOE of the promoted file for {year}.")
    return Solution(
        path=path,
        lcoe=layer["lcoe"],
        coverage=layer["coverage"],
        served_fraction=layer["served_fraction"],
        load_mw=layer["load_mw"],
        cost_of_capital=layer["cost_of_capital"],
        installation_cost={tech: layer[f"installation_cost_{tech}"] for tech in TECHS},
        horizon=int(attrs.get("investment_horizon_years", lcoe_attrs.get("investment_horizon_years"))),
    )


def lcoe_split(
    solution: Solution, opex_pct: dict[str, np.ndarray], lifetimes: dict[str, int], valid: np.ndarray
) -> dict[str, np.ndarray]:
    """
    The LCOE per technology in USD per MWh delivered; the parts add up to the LCOE. ``opex_pct`` is the fixed
    O&M share of CAPEX per technology and pixel, from the cost workbook of the run.
    """
    short = [tech for tech in TECHS if lifetimes[tech] < solution.horizon]
    if short:
        raise SolutionFileError(
            f"Lifetime of {', '.join(short)} is shorter than the {solution.horizon}-year horizon: BOA installs it "
            "again, which this split does not include."
        )
    w = np.where(valid, solution.cost_of_capital, 0.05)
    annuity_sum = sum((1.0 + w) ** -t for t in range(1, solution.horizon + 1))
    delivered = np.where(valid, solution.load_mw * HOURS_IN_YEAR * annuity_sum * solution.served_fraction, 1.0)
    with np.errstate(invalid="ignore"):
        split = {
            tech: np.where(valid, solution.installation_cost[tech] * (1.0 + opex_pct[tech] * annuity_sum) / delivered,
                           np.nan)
            for tech in TECHS
        }
    total = sum(split.values())
    if valid.any() and not np.allclose(total[valid], solution.lcoe[valid], rtol=SPLIT_RTOL):
        worst = float(np.nanmax(np.abs(total[valid] / solution.lcoe[valid] - 1)))
        raise SolutionFileError(
            f"Solar, wind and battery cost in {solution.path.name} do not add up to the LCOE (largest difference "
            f"{worst:.2%}); the cost workbook may not be the one of the run."
        )
    return split


def operation_layers(
    operation: str, coverage: np.ndarray, served: np.ndarray, valid: np.ndarray, load_factor: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load factor of the plants and the ratio of the electricity price to the LCOE, per pixel.

    The plants pay for all the energy the system delivers (``served``) and use the share ``CF`` of the hours
    at full load, so the price is ``LCOE * served / CF``.
    """
    if load_factor is not None:
        cf = np.full(served.shape, float(load_factor))
        over = valid & (cf > served)
        if over.any():
            raise ValueError(
                f"load_factor {load_factor} is more than the served fraction in {int(over.sum())} pixels "
                f"(lowest {float(np.nanmin(served[valid])):.4f}): the plants would use more energy than BOA delivers."
            )
    elif operation == FULL_LOAD:
        cf = coverage
    elif operation == PART_LOAD:
        cf = served
    else:
        raise ValueError(f"operation must be one of {', '.join(OPERATIONS)}, got {operation!r}.")
    cf = np.where(valid, cf, 1.0)
    return cf, np.where(valid, served, 1.0) / cf
