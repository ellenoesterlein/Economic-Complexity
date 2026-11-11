import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from boa.inputs.costs import COST_CACHE_VERSION

from lcom.chain import co2_cost, lcoh, lcom
from lcom.cli import main
from lcom.electricity import SolutionFileError
from lcom.flags import COST_WORKBOOK_FLAGS
from lcom.paper_inputs import route_b_params
from lcom.register import REGISTER_COLUMNS
from lcom.run import COST_WORKBOOK_NAME, LcoeFileError, file_sha256, run

COST_SET = "default"
RUN = "wy2025_test_ab12"
COVERAGE = 0.9
SERVED = 0.93
HORIZON = 25
OPEX = {"solar": 0.01, "wind": 0.02, "battery": 0.02}
SHARES = {"solar": 0.5, "wind": 0.3, "battery": 0.2}  # LCOE split of every pixel in the solution file
WACC = {"DEU": 0.055, "IND": 0.073, "CHN": 0.06, "GRL": 0.0877}
# Index = cost_key_id. Empty key, a province whose key is not in the costs (falls back to CHN), the
# global average marker, and Greenland.
LEGEND = ",DEU,IND,GRL,CHN:CN-GS,UNKNOWN:GLOBAL_AVG"
LCOE_NAME = "optimal_lcoe_wy2025_01MWkm2_cov0p9_2045_2050.nc"


def _cost_workbook(run_dir) -> str:
    path = run_dir / COST_WORKBOOK_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet = pd.DataFrame(
        [[f.region, f.subregion, f.tech, f.workbook_value] for f in COST_WORKBOOK_FLAGS],
        columns=["irena region", "subregion code", "tech", 2024],
    )
    with pd.ExcelWriter(path) as writer:
        sheet.to_excel(writer, sheet_name="RES CAPEX projections", index=False)
    return file_sha256(path)


def _cost_cache(run_dir, sha: str) -> None:
    keys = list(WACC)
    zeros = np.zeros((len(keys), 1))
    ds = xr.Dataset(
        coords={"iso3": keys, "year": [2050]},
        data_vars={
            "Capex solar": (("iso3", "year"), zeros + 1e6),
            "Capex wind": (("iso3", "year"), zeros + 1e6),
            "Capex battery": (("iso3", "year"), zeros + 1e5),
            **{f"Opex {tech}": (("iso3",), np.full(len(keys), value)) for tech, value in OPEX.items()},
            "Cost of capital": (("iso3",), np.array([WACC[k] for k in keys])),
        },
        attrs={"cost_cache_version": COST_CACHE_VERSION, "workbook_sha256": sha},
    )
    cache = run_dir / "cache_costs"
    cache.mkdir(parents=True)
    ds.to_netcdf(cache / "cost_of_renewables_2050_investment_year.nc")


def _sha(run_dir) -> str:
    return file_sha256(run_dir / COST_WORKBOOK_NAME)


# Row 0: no key, DEU, IND, GRL, province. Row 1: DEU (status 0), global average, IND (LCOE NaN), DEU, DEU.
KEY_ID = np.array([[0, 1, 2, 3, 4], [1, 5, 2, 1, 1]], dtype="int16")
LCOE_2050 = np.array([[np.nan, 60.0, 90.0, 80.0, 50.0], [np.nan, 70.0, np.nan, 40.0, 0.0]], dtype="float32")
LAT, LON = [10.0, 10.25], [0.0, 0.25, 0.5, 0.75, 1]
SOLUTION_NAME = "optimal_sol_wy2025_01MWkm2_cov0p9_GLOBAL_2050.nc"


def _solution_file(folder, name: str = SOLUTION_NAME, **attrs):
    """A BOA solution file for LCOE_2050: coverage COVERAGE, served SERVED and the LCOE split SHARES."""
    pixel_wacc = np.array([np.nan, WACC["DEU"], WACC["IND"], WACC["GRL"], WACC["CHN"], np.mean(list(WACC.values()))])
    w = pixel_wacc[KEY_ID]
    annuity_sum = sum((1.0 + w) ** -t for t in range(1, HORIZON + 1))
    lcoe = LCOE_2050.astype(float)
    finite = np.isfinite(lcoe)
    load = 2.0
    delivered = load * 8760 * annuity_sum * SERVED
    install = {
        tech: np.where(finite, SHARES[tech] * lcoe * delivered / (1 + OPEX[tech] * annuity_sum), np.nan)
        for tech in SHARES
    }
    ds = xr.Dataset(
        coords={"lat": LAT, "lon": LON},
        data_vars={
            "lcoe": (("lat", "lon"), LCOE_2050),
            "coverage": (("lat", "lon"), np.where(finite, COVERAGE, np.nan)),
            "served_fraction": (("lat", "lon"), np.where(finite, SERVED, np.nan)),
            "load_mw": (("lat", "lon"), np.full((2, 5), load)),
            "cost_of_capital": (("lat", "lon"), w),
            **{f"installation_cost_{tech}": (("lat", "lon"), value) for tech, value in install.items()},
        },
        attrs={
            "investment_year": 2050,
            "investment_horizon_years": HORIZON,
            "coverage_fraction": COVERAGE,
            "era5_weather_year": 2025,
            **attrs,
        },
    )
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return path


def _lcoe_file(run_dir, sha: str, layout_attrs: dict, name: str = LCOE_NAME, drop: tuple[str, ...] = ()):
    key_id = KEY_ID
    status = np.array([[0, 1, 1, 1, 1], [3, 1, 1, 1, 1]], dtype="int8")
    lcoe = np.array([np.full((2, 5), 99.0), LCOE_2050], dtype="float32")
    attrs = {
        "coverage_fraction": COVERAGE,
        "era5_weather_year": 2025,
        "cost_set": COST_SET,
        "run": RUN,
        "input_data_sha256": sha,
        "cost_key_legend": LEGEND,
        **layout_attrs,
    }
    ds = xr.Dataset(
        coords={"year": np.array([2045, 2050], dtype="int32"), "lat": LAT, "lon": LON},
        data_vars={
            "lcoe": (("year", "lat", "lon"), lcoe),
            "cost_key_id": (("lat", "lon"), key_id),
            "status": (("lat", "lon"), status),
        },
        attrs={k: v for k, v in attrs.items() if k not in drop},
    )
    path = run_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    _solution_file(run_dir)
    return path


SCHEMA_1 = {
    "schema_version": 1,
    "lcoe_basis": "per_mwh_delivered",
    "capacity_ceiling_applied": 0,
    "boa_version": "1.10.0",
}
LEGACY = {"boa_version": "1.8.0"}


@pytest.fixture
def run_dir(tmp_path):
    """The folder a user fills: data/boa/<run>/ with the LCOE file and the run's cost workbook."""
    run_dir = tmp_path / "data" / "boa" / RUN
    sha = _cost_workbook(run_dir)
    _cost_cache(run_dir, sha)
    return run_dir


@pytest.fixture
def lcoe_path(run_dir):
    return _lcoe_file(run_dir, _sha(run_dir), SCHEMA_1)


def _expected_lcom(lcoe: float, w: float, cf: float = COVERAGE) -> float:
    """USD/MWh_MeOH; the plants pay for all the energy delivered (SERVED) and use the share cf."""
    p = route_b_params()
    e = lcoe * SERVED / cf
    hydrogen = lcoh(e, w, cf, p.electrolysis)
    return lcom(e, w, cf, hydrogen, co2_cost(e, w, cf, p.dac, p.heat_pump), p.methanol).total


def _at(ds, name, lat, lon) -> float:
    return float(ds[name].isel(year=0, lat=lat, lon=lon))


def test_run_writes_result_register_and_provenance(lcoe_path, tmp_path):
    result = run(lcoe_path, tmp_path / "out")

    assert result.netcdf.name == "lcom_wy2025_01MWkm2_cov0p9_2045_2050_y2050.nc"
    assert result.netcdf.parent.name == RUN
    with xr.open_dataset(result.netcdf) as out:
        assert out["lcom_usd_per_t"].dims == ("year", "lat", "lon")
        assert out["year"].values.tolist() == [2050]
        assert out.attrs["lcoe_layout"] == "schema 1"
        assert out.attrs["operation"] == "full-load"
        assert out.attrs["solution_file"] == SOLUTION_NAME
        assert out.attrs["cost_key_legend"] == LEGEND
        for key in (
            "capex.china.solar",
            "capex.china.provinces",
            "wacc.country",
            "dac-weather-fixed",
            "heat-integration-not-modelled",
            "firm-power-design",
            "operation.full-load-only",
            "lcoe.capacity-ceiling-not-applied",
        ):
            assert key in out.attrs["data_flags"]
    provenance = json.loads(result.provenance.read_text(encoding="utf-8"))
    assert provenance["usd_per_eur"]["value"] == pytest.approx(1.08238046875)
    assert provenance["wacc"]["pixels_priced"] == 6
    assert provenance["operation"]["median"]["served_fraction"] == pytest.approx(SERVED)


def test_each_pixel_uses_its_lcoe_and_wacc(lcoe_path, tmp_path):
    result = run(lcoe_path, tmp_path / "out")

    with xr.open_dataset(result.netcdf) as out:
        assert _at(out, "wacc", 0, 1) == pytest.approx(0.055)
        assert _at(out, "lcom_usd_per_mwh", 0, 1) == pytest.approx(_expected_lcom(60.0, 0.055), rel=1e-6)
        assert _at(out, "lcom_usd_per_mwh", 0, 2) == pytest.approx(_expected_lcom(90.0, 0.073), rel=1e-6)
        parts = sum(out[f"lcom_{p}_usd_per_mwh"] for p in ("hydrogen", "electricity", "co2", "capital", "fom"))
        np.testing.assert_allclose(out["lcom_usd_per_mwh"].values, parts.values, rtol=1e-5)


def test_province_and_global_average_keys_follow_boa(lcoe_path, tmp_path):
    result = run(lcoe_path, tmp_path / "out")

    with xr.open_dataset(result.netcdf) as out:
        assert _at(out, "wacc", 0, 4) == pytest.approx(WACC["CHN"])  # CHN:CN-GS falls back to CHN
        assert _at(out, "wacc", 1, 1) == pytest.approx(np.mean(list(WACC.values())))  # UNKNOWN:GLOBAL_AVG


def test_pixels_without_a_result_stay_empty(lcoe_path, tmp_path):
    result = run(lcoe_path, tmp_path / "out")

    with xr.open_dataset(result.netcdf) as out:
        assert np.isnan(_at(out, "lcom_usd_per_mwh", 0, 0))  # empty key, status 0
        assert np.isnan(_at(out, "wacc", 0, 0))  # no global-average WACC for an empty key
        assert np.isnan(_at(out, "lcom_usd_per_mwh", 1, 0))  # status 3
        assert np.isnan(_at(out, "lcom_usd_per_mwh", 1, 2))  # LCOE NaN
        assert np.isnan(_at(out, "lcom_usd_per_mwh", 1, 4))  # LCOE 0 (legacy "no optimum")
        assert _at(out, "lcom_usd_per_mwh", 0, 3) == pytest.approx(_expected_lcom(80.0, 0.0877), rel=1e-6)  # GRL


def test_legacy_layout_is_accepted_and_recorded(run_dir, tmp_path):
    path = _lcoe_file(run_dir, _sha(run_dir), LEGACY)

    result = run(path, tmp_path / "out")

    with xr.open_dataset(result.netcdf) as out:
        assert out.attrs["lcoe_layout"] == "legacy BOA 1.8.0"
        assert out.attrs["capacity_ceiling_applied"] == "not recorded"
        assert "lcoe.capacity-ceiling-not-applied" in out.attrs["data_flags"]
        assert np.isfinite(_at(out, "lcom_usd_per_mwh", 0, 1))


def test_overrides_are_used_recorded_and_named(lcoe_path, tmp_path):
    result = run(lcoe_path, tmp_path / "out", load_factor=0.85, wacc=0.08)

    assert result.netcdf.name.endswith("_y2050_lf0p85_wacc0p08.nc")
    with xr.open_dataset(result.netcdf) as out:
        assert out.attrs["operation"] == "fixed load factor"
        assert "operation.fixed-load-factor" in out.attrs["data_flags"]
        assert "override" in out.attrs["wacc_basis"]
        assert _at(out, "wacc", 0, 2) == pytest.approx(0.08)
        assert _at(out, "lcom_usd_per_mwh", 0, 2) == pytest.approx(_expected_lcom(90.0, 0.08, 0.85), rel=1e-6)


def test_load_factor_above_the_served_fraction_is_refused(lcoe_path, tmp_path):
    with pytest.raises(ValueError, match="more than the served fraction in 6 pixels"):
        run(lcoe_path, tmp_path / "out", load_factor=0.95)


def test_part_load_uses_the_served_fraction_at_the_lcoe(lcoe_path, tmp_path):
    full = run(lcoe_path, tmp_path / "out")
    part = run(lcoe_path, tmp_path / "out", operation="part-load")

    assert part.netcdf.name.endswith("_y2050_partload.nc")
    with xr.open_dataset(part.netcdf) as out:
        assert out.attrs["operation"] == "part-load"
        assert "operation.part-load-no-minimum" in out.attrs["data_flags"]
        assert _at(out, "load_factor", 0, 1) == pytest.approx(SERVED)
        assert _at(out, "electricity_price", 0, 1) == pytest.approx(60.0)
        assert _at(out, "lcom_usd_per_mwh", 0, 1) == pytest.approx(_expected_lcom(60.0, 0.055, SERVED), rel=1e-6)
        part_bounds = (out["lcom_full_load_usd_per_t"].values, out["lcom_part_load_usd_per_t"].values)
    with xr.open_dataset(full.netcdf) as out:
        assert _at(out, "electricity_price", 0, 1) == pytest.approx(60.0 * SERVED / COVERAGE, rel=1e-6)
        np.testing.assert_allclose(out["lcom_part_load_usd_per_t"].values, part_bounds[1], rtol=1e-6)
        np.testing.assert_allclose(out["lcom_full_load_usd_per_t"].values, part_bounds[0], rtol=1e-6)
        np.testing.assert_allclose(out["lcom_usd_per_t"].values, part_bounds[0], rtol=1e-6)
        priced = np.isfinite(part_bounds[0])
        assert (part_bounds[1][priced] < part_bounds[0][priced]).all()


def test_cost_items_add_up_and_electricity_splits_like_the_lcoe(lcoe_path, tmp_path):
    result = run(lcoe_path, tmp_path / "out")

    with xr.open_dataset(result.netcdf) as out:
        items = ("electrolyser", "dac", "heat_pump", "synthesis") + tuple(
            f"electricity_{stage}" for stage in ("electrolysis", "dac", "synthesis")
        )
        total = sum(out[f"lcom_item_{item}_usd_per_t"] for item in items)
        np.testing.assert_allclose(total.values, out["lcom_usd_per_t"].values, rtol=1e-5)
        electricity = sum(out[f"lcom_item_electricity_{stage}_usd_per_t"] for stage in ("electrolysis", "dac", "synthesis"))
        for tech, share in SHARES.items():
            np.testing.assert_allclose(out[f"lcoe_{tech}"].values, share * out["lcoe"].values, rtol=1e-5)
            np.testing.assert_allclose(
                out[f"lcom_item_electricity_{tech}_usd_per_t"].values, share * electricity.values, rtol=1e-5
            )


@pytest.mark.parametrize(
    ("attrs", "match"),
    [
        ({"investment_year": 2045}, "investment year 2045"),
        ({"coverage_fraction": 0.85}, "coverage_fraction"),
    ],
)
def test_solution_file_of_another_run_is_refused(lcoe_path, tmp_path, attrs, match):
    _solution_file(lcoe_path.parent, **attrs)

    with pytest.raises(SolutionFileError, match=match):
        run(lcoe_path, tmp_path / "out")


def test_solution_file_with_another_lcoe_is_refused(lcoe_path, tmp_path):
    path = _solution_file(tmp_path / "other")
    with xr.open_dataset(path) as ds:
        changed = ds.load().assign(lcoe=ds["lcoe"] * 1.01)
    changed.to_netcdf(lcoe_path.parent / SOLUTION_NAME)

    with pytest.raises(SolutionFileError, match="not the LCOE of the promoted file"):
        run(lcoe_path, tmp_path / "out")


def test_split_that_does_not_add_up_is_refused(lcoe_path, tmp_path):
    path = lcoe_path.parent / SOLUTION_NAME
    with xr.open_dataset(path) as ds:
        changed = ds.load().assign(installation_cost_wind=ds["installation_cost_wind"] * 1.1)
    changed.to_netcdf(path)

    with pytest.raises(SolutionFileError, match="do not add up to the LCOE"):
        run(lcoe_path, tmp_path / "out")


def test_missing_solution_file_is_refused(lcoe_path, tmp_path):
    (lcoe_path.parent / SOLUTION_NAME).unlink()

    with pytest.raises(SolutionFileError, match="No BOA solution file"):
        run(lcoe_path, tmp_path / "out")
    other = _solution_file(tmp_path / "elsewhere", name="optimal_sol_x_GLOBAL_2050.nc")
    assert run(lcoe_path, tmp_path / "out", solution_path=other).netcdf.exists()


def test_lcoe_with_the_capacity_ceiling_carries_no_ceiling_flag(run_dir, tmp_path):
    path = _lcoe_file(run_dir, _sha(run_dir), {**SCHEMA_1, "capacity_ceiling_applied": 1})

    result = run(path, tmp_path / "out")

    with xr.open_dataset(result.netcdf) as out:
        assert "lcoe.capacity-ceiling-not-applied" not in out.attrs["data_flags"]
    provenance = json.loads(result.provenance.read_text(encoding="utf-8"))
    assert "lcoe.capacity-ceiling-not-applied" not in [f["key"] for f in provenance["data_flags"]]


@pytest.mark.parametrize(
    ("attrs", "match"),
    [
        ({**SCHEMA_1, "schema_version": 2}, "schema_version 2"),
        ({**SCHEMA_1, "lcoe_basis": "per_mwh_generated"}, "lcoe_basis"),
        ({"boa_version": "1.7.0"}, "BOA 1.7.0"),
    ],
)
def test_unknown_layouts_are_refused(run_dir, tmp_path, attrs, match):
    path = _lcoe_file(run_dir, _sha(run_dir), attrs)

    with pytest.raises(LcoeFileError, match=match):
        run(path, tmp_path / "out")


def test_other_workbook_is_refused(run_dir, tmp_path):
    other = _lcoe_file(run_dir, "0" * 64, SCHEMA_1, name="optimal_lcoe_other.nc")

    with pytest.raises(LcoeFileError, match="not the one"):
        run(other, tmp_path / "out")


def test_lcoe_file_without_a_workbook_next_to_it_is_refused(run_dir, tmp_path):
    alone = _lcoe_file(tmp_path / "elsewhere", _sha(run_dir), SCHEMA_1)

    with pytest.raises(LcoeFileError, match=f"No cost workbook.*'{COST_SET}'"):
        run(alone, tmp_path / "out")


def test_cost_workbook_can_be_given_explicitly(run_dir, tmp_path):
    alone = _lcoe_file(tmp_path / "elsewhere", _sha(run_dir), SCHEMA_1)

    result = run(alone, tmp_path / "out", cost_workbook=run_dir / COST_WORKBOOK_NAME)

    with xr.open_dataset(result.netcdf) as out:
        assert _at(out, "wacc", 0, 1) == pytest.approx(0.055)


@pytest.mark.parametrize("attr", ["coverage_fraction", "cost_set", "cost_key_legend"])
def test_lcoe_file_without_a_required_attribute_is_refused(run_dir, tmp_path, attr):
    path = _lcoe_file(run_dir, _sha(run_dir), SCHEMA_1, drop=(attr,))

    with pytest.raises(LcoeFileError, match=f"not a promoted LCOE file \\(missing {attr}\\)"):
        run(path, tmp_path / "out")


def test_non_promoted_file_is_refused(tmp_path):
    path = tmp_path / "x.nc"
    xr.Dataset({"lcoe": (("lat",), [1.0])}).to_netcdf(path)

    with pytest.raises(LcoeFileError, match="not a promoted LCOE file"):
        run(path, tmp_path / "out")


def test_register_has_a_source_for_every_row(lcoe_path, tmp_path):
    result = run(lcoe_path, tmp_path / "out")

    register = pd.read_csv(result.register)
    assert list(register.columns) == REGISTER_COLUMNS
    assert register["source"].notna().all() and (register["source"].str.strip() != "").all()
    assert {"electrolysis", "methanol", "dac", "heat_pump", "currency", "boa"} == set(register["stage"])


def test_cli_returns_zero_on_success_and_one_on_a_refused_file(run_dir, lcoe_path, tmp_path):
    out = tmp_path / "out"
    assert main([str(lcoe_path), "--out-dir", str(out)]) == 0
    bad = _lcoe_file(run_dir, _sha(run_dir), {"boa_version": "1.7.0"}, name="optimal_lcoe_bad.nc")
    assert main([str(bad), "--out-dir", str(out)]) == 1
    alone = _lcoe_file(tmp_path / "elsewhere", _sha(run_dir), SCHEMA_1)
    assert main([str(alone), "--out-dir", str(out), "--cost-workbook", str(run_dir / COST_WORKBOOK_NAME)]) == 0
