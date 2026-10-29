import numpy as np
import pytest
import xarray as xr

from lcom.compare import FIG8C_ROUTE_B_EUR_PER_T, compare, main, paper_lcom_by_region, result_lcom_by_pixel
from lcom.paper_inputs import USD_PER_EUR

# EUR/t per region; the file stores 1e-6 of these values.
PAPER = {"USA.51.4_1": 658.0, "USA.1.1_1": 700.0, "IND.34.18_1": 1132.0, "IND.2.2_1": 900.0, "CHN.26.16_1": 1024.0}


def _paper_csv(path):
    regions = [*PAPER, "AFG.1.1_1"]
    lcom = [f"{v * 1e-6!r}" for v in PAPER.values()] + [""]  # AFG has no result
    lines = [
        "," + ",".join(regions),
        "LCO_MeOH," + ",".join(lcom),
        "Prod_MeOH," + ",".join("1.0" for _ in regions),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _result_nc(path):
    # Legend: empty, USA, IND, a CHN province, DEU (not a paper country).
    legend = ",USA,IND,CHN:CN-GS,DEU"
    key_id = np.array([[1, 1, 2, 3, 4, 0]], dtype="int16")
    eur = np.array([[600.0, 800.0, 1000.0, 900.0, 950.0, np.nan]])
    ds = xr.Dataset(
        {
            "lcom_usd_per_t": (("year", "lat", "lon"), (eur * USD_PER_EUR)[np.newaxis]),
            "cost_key_id": (("lat", "lon"), key_id),
        },
        coords={"year": [2050], "lat": [0.0], "lon": np.arange(6) * 0.25},
        attrs={"cost_key_legend": legend},
    )
    ds.to_netcdf(path)
    return path


def test_paper_values_are_scaled_to_eur_per_tonne(tmp_path):
    paper = paper_lcom_by_region(_paper_csv(tmp_path / "paper.csv"))

    assert paper["USA.51.4_1"] == pytest.approx(658.0)
    assert "AFG.1.1_1" not in paper.index


def test_result_pixels_carry_their_country(tmp_path):
    ours = result_lcom_by_pixel(_result_nc(tmp_path / "r.nc"))

    assert list(ours["iso3"]) == ["USA", "USA", "IND", "CHN", "DEU"]
    assert ours["eur_per_t"].iloc[0] == pytest.approx(600.0)


def test_comparison_keeps_the_paper_countries_only(tmp_path):
    paper = paper_lcom_by_region(_paper_csv(tmp_path / "paper.csv"))
    result = compare(paper, result_lcom_by_pixel(_result_nc(tmp_path / "r.nc")))

    countries = result.countries
    assert "DEU" not in countries.index
    assert countries.loc["USA", ("result", "median")] == pytest.approx(700.0)
    assert countries.loc["USA", ("paper_soec", "median")] == pytest.approx(679.0)
    assert countries.loc["IND", ("ratio", "median")] == pytest.approx(1000.0 / 1016.0)
    assert result.summary["countries_compared"] == 3
    assert result.summary["result_median"] == pytest.approx(850.0)  # USA 600, 800; IND 1000; CHN 900


def test_fig8c_table_lists_the_nine_regions(tmp_path):
    paper = paper_lcom_by_region(_paper_csv(tmp_path / "paper.csv"))
    fig8c = compare(paper, result_lcom_by_pixel(_result_nc(tmp_path / "r.nc"))).fig8c

    assert list(fig8c.index) == list(FIG8C_ROUTE_B_EUR_PER_T)
    assert fig8c.loc["IND.34.18_1", "paper_soec"] == pytest.approx(1132.0)
    assert fig8c.loc["CHN.26.16_1", "result_country_median"] == pytest.approx(900.0)
    assert np.isnan(fig8c.loc["MEX.16.24_1", "result_country_median"])


def test_cli_writes_both_tables(tmp_path, capsys):
    result = _result_nc(tmp_path / "r.nc")

    assert main([str(result), "--paper-results", str(_paper_csv(tmp_path / "paper.csv"))]) == 0
    assert (tmp_path / "r_compare_countries.csv").exists()
    assert (tmp_path / "r_compare_fig8c.csv").exists()
    assert "spearman_country_medians" in capsys.readouterr().out
