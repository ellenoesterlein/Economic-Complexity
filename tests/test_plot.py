import numpy as np
import pytest
import xarray as xr

from lcom.paper_inputs import USD_PER_EUR
from lcom.plot import border_segments, class_breaks, country_stats, electricity_share, load, main

# 6 x 8 grid: ocean in the first column, DEU in the west, a CHN province and CHN in the east, one DEU pixel
# without a result.
LEGEND = ",DEU,CHN:CN-GS,CHN"


def _result(path, seed: int = 0):
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


def test_cli_writes_four_plots_and_refuses_bad_breaks(tmp_path, capsys):
    path = _result(tmp_path / "lcom_test.nc")
    out = tmp_path / "plots"

    assert main([str(path), "--out-dir", str(out), "--top", "2"]) == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == [f"lcom_test_{kind}.png" for kind in
                     ("countries", "distribution", "map_electricity_share", "map_lcom")]
    assert main([str(path), "--out-dir", str(out), "--breaks", "900", "800"]) == 1
    assert "breaks must be" in capsys.readouterr().err
