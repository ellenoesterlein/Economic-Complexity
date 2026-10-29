"""
Compare an LCOM result with Wenzel et al. (2026).

The paper's results per region are in its figshare data
(``processedYearlyResults_Year2018_Demand10_MeOH_1.csv``, doi:10.6084/m9.figshare.30664253): the SOEC
route (Route A), 2050, 10 % of the maximum technical potential, 20,265 GADM admin-2 regions in 78
water-stressed countries. That file is the source of the paper's headline numbers (median 976 EUR/t).
Route B values exist only for the nine regions of the paper's Fig. 8c, as a bar chart.

The comparison:

- per country, for the 78 countries of the paper: median, P10 and minimum LCOM of this result
  (pixels) next to the paper's (regions), in EUR/t;
- the median over all pixels of these 78 countries next to the paper's median;
- the Spearman rank correlation of the country medians;
- the nine Fig. 8c countries: this result's country median against the paper's Route B value of
  one region in that country, ranked.

Pixels and admin-2 regions are different units: a pixel median weights a country by area, a region
median by its number of regions.

Usage:
    python -m lcom.compare RESULT.nc [RESULT.nc ...] --paper-results <figshare>/processedYearlyResults_Year2018_Demand10_MeOH_1.csv
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import spearmanr

from lcom.paper_inputs import USD_PER_EUR

# The figshare results give LCO_MeOH in units of 1e6 EUR/t.
PAPER_LCOM_SCALE = 1e6
PAPER_MEDIAN_EUR_PER_T = 976.0

# Route B (PEM) LCOM of the nine regions in the paper's Fig. 8c, EUR/t, read by eye from the bar chart
# (about +-10 EUR/t). The figshare data has no Route B results.
FIG8C_ROUTE_B_EUR_PER_T: dict[str, float] = {
    "USA.51.4_1": 570.0,
    "MEX.16.24_1": 850.0,
    "ARG.20.6_1": 515.0,
    "FRA.13.5_1": 840.0,
    "SAU.13_1": 715.0,
    "ZAF.9.6_1": 675.0,
    "IND.34.18_1": 1020.0,
    "CHN.26.16_1": 900.0,
    "AUS.10.19_1": 740.0,
}


def paper_lcom_by_region(path: Path) -> pd.Series:
    """The paper's LCOM per region in EUR/t. The file is transposed: one column per region, one row per metric."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = {row[0]: row for row in csv.reader(f)}
    if "LCO_MeOH" not in rows:
        raise ValueError(f"{path.name} has no LCO_MeOH row.")
    lcom = pd.to_numeric(pd.Series(rows["LCO_MeOH"][1:], index=rows[""][1:]), errors="coerce") * PAPER_LCOM_SCALE
    return lcom.dropna().rename("eur_per_t")


def result_lcom_by_pixel(path: Path) -> pd.DataFrame:
    """This result's LCOM per priced pixel in EUR/t, with the pixel's country (ISO3)."""
    with xr.open_dataset(path) as ds:
        usd = ds["lcom_usd_per_t"].isel(year=0).values
        key_id = ds["cost_key_id"].values
        legend = np.array([key.split(":")[0] for key in str(ds.attrs["cost_key_legend"]).split(",")])
    priced = np.isfinite(usd)
    return pd.DataFrame({"iso3": legend[key_id[priced]], "eur_per_t": usd[priced] / USD_PER_EUR})


def _stats(values: pd.core.groupby.SeriesGroupBy) -> pd.DataFrame:
    return pd.DataFrame(
        {"n": values.size(), "median": values.median(), "p10": values.quantile(0.1), "min": values.min()}
    )


@dataclass(frozen=True)
class Comparison:
    countries: pd.DataFrame  # per country of the paper: result and paper statistics, EUR/t
    fig8c: pd.DataFrame  # the nine Fig. 8c regions
    summary: dict[str, float]


def compare(paper: pd.Series, ours: pd.DataFrame) -> Comparison:
    """
    Args:
        paper: ``paper_lcom_by_region`` of the Demand10 results.
        ours: ``result_lcom_by_pixel`` of one result.
    """
    paper_iso3 = paper.index.str.split(".").str[0]
    ours = ours[ours["iso3"].isin(set(paper_iso3))]
    countries = pd.concat(
        {
            "result": _stats(ours.groupby("iso3")["eur_per_t"]),
            "paper_soec": _stats(paper.groupby(paper_iso3)),
        },
        axis=1,
    )
    countries[("ratio", "median")] = countries[("result", "median")] / countries[("paper_soec", "median")]
    countries.index.name = "iso3"

    medians = countries[("result", "median")]
    fig8c = pd.DataFrame(
        [
            {
                "region": region,
                "iso3": region.split(".")[0],
                "paper_soec": paper.get(region, np.nan),
                "paper_route_b_read_by_eye": route_b,
                "result_country_median": medians.get(region.split(".")[0], np.nan),
            }
            for region, route_b in FIG8C_ROUTE_B_EUR_PER_T.items()
        ]
    ).set_index("region")
    fig8c["rank_paper_route_b"] = fig8c["paper_route_b_read_by_eye"].rank()
    fig8c["rank_result"] = fig8c["result_country_median"].rank()

    both = countries.dropna(subset=[("result", "median"), ("paper_soec", "median")])
    summary = {
        "countries_compared": float(len(both)),
        "result_median": float(ours["eur_per_t"].median()),
        "result_p10": float(ours["eur_per_t"].quantile(0.1)),
        "paper_soec_median": float(paper.median()),
        "spearman_country_medians": float(
            spearmanr(both[("result", "median")], both[("paper_soec", "median")]).statistic
        ),
        "spearman_fig8c_route_b": float(
            spearmanr(fig8c["result_country_median"], fig8c["paper_route_b_read_by_eye"]).statistic
        ),
    }
    return Comparison(countries, fig8c, summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lcom-compare", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", nargs="+", type=Path, help="LCOM result files (lcom_*.nc).")
    parser.add_argument("--paper-results", type=Path, required=True, help="The paper's Demand10 results CSV.")
    args = parser.parse_args(argv)
    paper = paper_lcom_by_region(args.paper_results)
    for result in args.results:
        comparison = compare(paper, result_lcom_by_pixel(result))
        comparison.countries.to_csv(result.with_name(f"{result.stem}_compare_countries.csv"))
        comparison.fig8c.to_csv(result.with_name(f"{result.stem}_compare_fig8c.csv"))
        print(f"== {result.name}")
        for key, value in comparison.summary.items():
            print(f"  {key}: {value:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
