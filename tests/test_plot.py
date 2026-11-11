import numpy as np
import pandas as pd
import pytest
import xarray as xr

from lcom.paper_inputs import USD_PER_EUR
from lcom.plot import (
    BREAKDOWN_ITEMS,
    border_segments,
    breakdown_groups,
    breakdown_stats,
    class_breaks,
    country_stats,
    electricity_share,
    load,
    load_items,
    main,
)

# 6 x 8 grid: ocean in the first column, DEU in the west, a CHN province and CHN in the east, one DEU pixel
# without a result.
LEGEND = ",DEU,CHN:CN-GS,CHN"


# Item shares of the LCOM per tonne in the test result; they add up to 1 without the electricity parts' sum.
ITEM_SHARES = {
    "electricity_solar": 0.4,
    "electricity_wind": 0.2,
    "electricity_battery": 0.15,
    "electrolyser": 0.1,
    "dac": 0.08,
    "heat_pump": 0.03,
    "synthesis": 0.04,
}


def _result(path, seed: int = 0, items: bool = True):
    rng = np.random.default_rng(seed)
    key_id = np.zeros((6, 8), dtype="int16")
    key_id[:, 1:4] = 1
    key_id[:3, 4:] = 2
    key_id[3:, 4:] = 3
    lcom_mwh = rng.uniform(100, 250, (6, 8))
    lcom_mwh[key_id == 0] = np.nan
    lcom_mwh[0, 1] = np.nan
    # Split: hydrogen 60 %, CO2 30 %, synthesis electricity 4 %, rest capital and FOM.
    layers = {
        "lcom_usd_per_mwh": lcom_mwh,
        "lcom_usd_per_t": lcom_mwh * 5.528,
        "lcom_hydrogen_usd_per_mwh": lcom_mwh * 0.6,
        "lcom_co2_usd_per_mwh": lcom_mwh * 0.3,
        "lcom_electricity_usd_per_mwh": lcom_mwh * 0.04,
        "lcoh_usd_per_mwh": np.full((6, 8), 100.0),
        "lcoh_electricity_usd_per_mwh": np.full((6, 8), 80.0),
        "co2_cost_usd_per_t": np.full((6, 8), 200.0),
        "co2_electricity_usd_per_t": np.full((6, 8), 100.0),
    }
    if items:
        layers.update({f"lcom_item_{item}_usd_per_t": lcom_mwh * 5.528 * share for item, share in ITEM_SHARES.items()})
    ds = xr.Dataset(
        coords={"year": [2050], "lat": np.arange(6) * 0.25 + 10, "lon": np.arange(8) * 0.25},
        data_vars={
            **{name: (("year", "lat", "lon"), value[np.newaxis].astype("float32")) for name, value in layers.items()},
            "cost_key_id": (("lat", "lon"), key_id),
        },
        attrs={"cost_key_legend": LEGEND, "run": "wy2025_test"},
    )
    ds.to_netcdf(path)
    return path


def test_electricity_share_adds_the_electricity_of_every_stage(tmp_path):
    with xr.open_dataset(_result(tmp_path / "lcom.nc")) as ds:
        share = electricity_share(ds)

    # 0.6 * 80/100 + 0.3 * 100/200 + 0.04
    assert np.nanmax(np.abs(share - (0.48 + 0.15 + 0.04))) < 1e-6
    assert np.isnan(share[0, 0])


def test_load_converts_to_eur_and_maps_provinces_to_their_country(tmp_path):
    path = _result(tmp_path / "lcom.nc")

    eur, usd = load(path, "EUR"), load(path, "USD")

    np.testing.assert_allclose(eur.lcom * USD_PER_EUR, usd.lcom, rtol=1e-6)
    assert set(np.unique(eur.iso3)) == {"", "DEU", "CHN"}


def test_border_segments_follow_country_edges_and_coasts():
    iso3 = np.array([["", "A", "B"], ["", "A", "A"]])

    segments = border_segments(iso3, np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0]))

    # Vertical: ""|A twice, A|B once; horizontal: B over A once.
    assert len(segments) == 4
    assert [[1.5, -0.5], [1.5, 0.5]] in segments.tolist()


def test_country_stats_select_and_sort(tmp_path):
    result = load(_result(tmp_path / "lcom.nc"))

    top = country_stats(result, top=1)
    chosen = country_stats(result, ["DEU", "CHN"])

    assert list(top.index) == ["CHN"]  # 24 priced pixels (province included) against 17
    assert list(chosen.index) == sorted(["DEU", "CHN"], key=lambda c: chosen.loc[c, "median"])
    assert chosen.loc["DEU", "n"] == 17
    assert (chosen["p10"] <= chosen["median"]).all() and (chosen["median"] <= chosen["p90"]).all()
    with pytest.raises(ValueError, match="No priced pixels for FRA"):
        country_stats(result, ["FRA"])


def test_class_breaks_are_rounded_deciles():
    breaks = class_breaks(np.arange(400.0, 1400.0))

    assert breaks[0] == 500 and breaks[-1] == 1300
    assert np.all(np.diff(breaks) > 0)


def test_breakdown_takes_the_cheapest_share_of_each_country(tmp_path):
    items = load_items(_result(tmp_path / "lcom.nc"), "USD")

    stats = breakdown_stats(items, ["DEU", "CHN"], share=0.2)

    # DEU: 17 priced pixels -> the 4 cheapest (ceil of 3.4); CHN: 24 -> 5.
    assert stats.loc["DEU", "n"] == 4 and stats.loc["CHN", "n"] == 5
    deu = items[items["iso3"] == "DEU"].nsmallest(4, "lcom")
    assert stats.loc["DEU", "lcom_mean"] == pytest.approx(deu["lcom"].mean())
    assert sum(stats.loc["DEU", f"{item}_mean"] for item, _ in BREAKDOWN_ITEMS) == pytest.approx(
        deu["lcom"].mean(), rel=1e-5
    )
    assert (stats["electricity_solar_p10"] <= stats["electricity_solar_p90"]).all()
    assert list(stats.index) == sorted(["DEU", "CHN"], key=lambda c: stats.loc[c, "lcom_mean"])
    with pytest.raises(ValueError, match="No priced pixels for FRA"):
        breakdown_stats(items, ["FRA"])


def test_breakdown_groups_cheapest_interest_and_most_expensive(capsys):
    stats = pd.DataFrame(
        {"pixels": [100, 5, 50, 30, 200], "lcom_mean": [400.0, 300.0, 600.0, 900.0, 1200.0]},
        index=pd.Index(["AAA", "TTO", "CCC", "DDD", "EEE"], name="iso3"),
    ).sort_values("lcom_mean")

    groups = dict(breakdown_groups(stats, cheapest=2, interest=["TTO", "CCC", "XXX"], expensive=1, min_pixels=20))

    assert list(groups) == ["Cheapest 2", "Countries of interest", "Most expensive 1"]
    assert list(groups["Cheapest 2"].index) == ["AAA", "CCC"]  # TTO is cheaper but has 5 pixels
    assert list(groups["Countries of interest"].index) == ["TTO", "CCC"]
    assert list(groups["Most expensive 1"].index) == ["EEE"]
    assert "no priced pixels for XXX" in capsys.readouterr().err
    assert [title for title, _ in breakdown_groups(stats, 0, [], 0)] == []


def test_result_without_items_has_no_breakdown(tmp_path):
    with pytest.raises(ValueError, match="no cost items"):
        load_items(_result(tmp_path / "lcom.nc", items=False))


def test_cli_writes_the_plots_and_refuses_bad_breaks(tmp_path, capsys):
    path = _result(tmp_path / "lcom_test.nc")
    out = tmp_path / "plots"

    assert main([str(path), "--out-dir", str(out), "--top", "2", "--interest", "DEU"]) == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == sorted(
        [f"lcom_test_{kind}.png" for kind in ("countries", "distribution", "map_electricity_share", "map_lcom")]
        + ["lcom_test_breakdown_top5.png", "lcom_test_breakdown_top5.csv"]
    )
    table = pd.read_csv(out / "lcom_test_breakdown_top5.csv", index_col="iso3")
    assert set(table.index) == {"DEU", "CHN"}  # every country, not only the plotted ones
    without = _result(tmp_path / "lcom_old.nc", items=False)
    assert main([str(without), "--out-dir", str(out), "--top", "2"]) == 0
    assert "no breakdown plot" in capsys.readouterr().err
    assert not (out / "lcom_old_breakdown_top5.png").exists()
    assert main([str(path), "--out-dir", str(out), "--breaks", "900", "800"]) == 1
    assert "breaks must be" in capsys.readouterr().err
