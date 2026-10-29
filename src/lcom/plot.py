"""
Plot an LCOM result: two world maps, the distribution over pixels and the country medians.

For each result file (``lcom_*.nc``) it writes four PNG files next to it (or to ``--out-dir``):

- ``<stem>_map_lcom.png``: LCOM per pixel in 10 classes, with the deciles of the priced pixels as class
  limits, or ``--breaks``; give the same breaks to compare runs.
- ``<stem>_map_electricity_share.png``: the share of the LCOM that is electricity (electrolysis, DAC, heat
  pump and synthesis plant), in decile classes.
- ``<stem>_distribution.png``: histogram of the LCOM over the priced pixels, with the median and the P10.
- ``<stem>_countries.png``: median and P10 to P90 per country, for the countries with the most priced pixels
  or for ``--countries``.

Borders are the edges between the cost keys of the result file (country level, 0.25 degree steps), so no
shapefile is necessary. Grey land has a cost key but no result.

Examples:
    python -m lcom.plot outputs/<run>/lcom_*.nc
    python -m lcom.plot RESULT.nc --currency USD --breaks 500 600 700 800 900 1000 1200 1500
    python -m lcom.plot RESULT.nc --countries CHN,IND,SAU,AUS,USA,ARG,MEX,ZAF,FRA
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import font_manager, rc_context
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.figure import Figure

from lcom.paper_inputs import USD_PER_EUR

# Reference palette of the plots: one blue sequential ramp (light = low), ink and surface tokens.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SERIES = "#2a78d6"
SERIES_LIGHT = "#86b6ef"
SURFACE = "#fcfcfb"
NO_RESULT = "#d9d8d2"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BORDER = "#7d7c77"
FONTS = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]
DPI = 200


@dataclass(frozen=True)
class Result:
    """One LCOM result, reduced to what the plots need. Costs are per tonne of methanol in ``currency``."""

    name: str
    currency: str
    lat: np.ndarray
    lon: np.ndarray
    lcom: np.ndarray  # (lat, lon); NaN where not priced
    electricity_share: np.ndarray  # (lat, lon), fraction of LCOM; NaN where not priced
    iso3: np.ndarray  # (lat, lon) str; "" outside every country
    run: str
    priced_year: int


def electricity_share(ds: xr.Dataset) -> np.ndarray:
    """Share of the LCOM that is electricity: in the hydrogen, in the CO2 and in the synthesis plant."""
    names = (
        "lcom_usd_per_mwh", "lcom_hydrogen_usd_per_mwh", "lcom_co2_usd_per_mwh", "lcom_electricity_usd_per_mwh",
        "lcoh_usd_per_mwh", "lcoh_electricity_usd_per_mwh", "co2_cost_usd_per_t", "co2_electricity_usd_per_t",
    )
    layer = {name: ds[name].isel(year=0).values.astype(float) for name in names}
    with np.errstate(invalid="ignore", divide="ignore"):
        electricity = (
            layer["lcom_hydrogen_usd_per_mwh"] * layer["lcoh_electricity_usd_per_mwh"] / layer["lcoh_usd_per_mwh"]
            + layer["lcom_co2_usd_per_mwh"] * layer["co2_electricity_usd_per_t"] / layer["co2_cost_usd_per_t"]
            + layer["lcom_electricity_usd_per_mwh"]
        )
        return electricity / layer["lcom_usd_per_mwh"]


def load(path: Path, currency: str = "EUR") -> Result:
    if currency not in ("EUR", "USD"):
        raise ValueError(f"currency must be EUR or USD, got {currency}.")
    with xr.open_dataset(path) as ds:
        usd = ds["lcom_usd_per_t"].isel(year=0).values.astype(float)
        legend = np.array([key.split(":")[0] for key in str(ds.attrs["cost_key_legend"]).split(",")])
        iso3 = legend[ds["cost_key_id"].values]
        iso3 = np.where(iso3 == "UNKNOWN", "", iso3)
        return Result(
            name=path.stem,
            currency=currency,
            lat=ds["lat"].values,
            lon=ds["lon"].values,
            lcom=usd / USD_PER_EUR if currency == "EUR" else usd,
            electricity_share=electricity_share(ds),
            iso3=iso3,
            run=str(ds.attrs.get("run", "")),
            priced_year=int(ds["year"].values[0]),
        )


def border_segments(iso3: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Pixel edges between two different country keys (coasts included), as (n, 2, 2) line segments."""
    dlat, dlon = abs(float(lat[1] - lat[0])), abs(float(lon[1] - lon[0]))
    rows, cols = np.nonzero(iso3[:, 1:] != iso3[:, :-1])  # vertical edges
    x = (lon[cols] + lon[cols + 1]) / 2
    vertical = np.stack([np.stack([x, lat[rows] - dlat / 2], 1), np.stack([x, lat[rows] + dlat / 2], 1)], 1)
    rows, cols = np.nonzero(iso3[1:, :] != iso3[:-1, :])  # horizontal edges
    y = (lat[rows] + lat[rows + 1]) / 2
    horizontal = np.stack([np.stack([lon[cols] - dlon / 2, y], 1), np.stack([lon[cols] + dlon / 2, y], 1)], 1)
    return np.concatenate([vertical, horizontal]).reshape(-1, 2, 2)


def country_stats(result: Result, countries: list[str] | None = None, top: int = 30) -> pd.DataFrame:
    """Median, P10, P90 and pixel count per country, sorted by median; the ``top`` countries by pixel count."""
    priced = np.isfinite(result.lcom) & (result.iso3 != "")
    df = pd.DataFrame({"iso3": result.iso3[priced], "lcom": result.lcom[priced]})
    stats = df.groupby("iso3")["lcom"].agg(
        n="size", median="median", p10=lambda v: v.quantile(0.1), p90=lambda v: v.quantile(0.9)
    )
    if countries:
        missing = sorted(set(countries) - set(stats.index))
        if missing:
            raise ValueError(f"No priced pixels for {', '.join(missing)}.")
        stats = stats.loc[countries]
    else:
        stats = stats.nlargest(top, "n")
    return stats.sort_values("median")


def _font() -> str:
    """The first font of FONTS that is installed; matplotlib warns for every missing family in a list."""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    return next((font for font in FONTS if font in installed), "DejaVu Sans")


def _figure(width: float, height: float) -> Figure:
    fig = Figure(figsize=(width, height), facecolor=SURFACE)
    fig.set_layout_engine("constrained")
    return fig


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=8, length=3, width=0.8)


def _titles(ax, title: str, subtitle: str) -> None:
    """Title and subtitle, left-aligned above the plot area."""
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=20)
    ax.annotate(subtitle, (0, 1), xycoords="axes fraction", xytext=(0, 6), textcoords="offset points",
                ha="left", va="bottom", fontsize=8.5, color=INK_SECONDARY)


def class_breaks(values: np.ndarray) -> np.ndarray:
    """Class limits of a map: the deciles of ``values``, rounded to two significant steps of their range."""
    deciles = np.quantile(values[np.isfinite(values)], np.linspace(0.1, 0.9, 9))
    spread = float(np.ptp(deciles)) or 1.0
    step = 10 ** np.floor(np.log10(spread)) / 10
    return np.unique(np.round(deciles / step) * step)


def plot_map(result: Result, values: np.ndarray, title: str, label: str, breaks: np.ndarray, path: Path) -> None:
    """A classed map: one ramp colour per class between ``breaks``, plus one class below and one above."""
    land = result.iso3 != ""
    rows, cols = np.nonzero(land.any(axis=1))[0], np.nonzero(land.any(axis=0))[0]
    window = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
    lat, lon = result.lat[window[0]], result.lon[window[1]]
    iso3, values = result.iso3[window], values[window]
    dlat, dlon = abs(float(lat[1] - lat[0])), abs(float(lon[1] - lon[0]))
    extent = (lon.min() - dlon / 2, lon.max() + dlon / 2, lat.min() - dlat / 2, lat.max() + dlat / 2)
    origin = "lower" if lat[0] < lat[-1] else "upper"
    width = 12.0
    fig = _figure(width, width * (extent[3] - extent[2]) / (extent[1] - extent[0]) + 1.4)
    ax = fig.add_subplot()
    ax.set_facecolor(SURFACE)
    ax.set_axis_off()

    land_without_result = np.where((iso3 != "") & ~np.isfinite(values), 1.0, np.nan)
    ax.imshow(land_without_result, extent=extent, origin=origin, cmap=ListedColormap([NO_RESULT]),
              interpolation="nearest")
    colors = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)(np.linspace(0, 1, len(breaks) + 1))
    cmap = ListedColormap(colors[1:-1]).with_extremes(under=colors[0], over=colors[-1], bad=(0, 0, 0, 0))
    image = ax.imshow(values, extent=extent, origin=origin, cmap=cmap, norm=BoundaryNorm(breaks, len(breaks) - 1),
                      interpolation="nearest")
    ax.add_collection(LineCollection(border_segments(iso3, lat, lon), colors=BORDER, linewidths=0.25))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    bar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.04, pad=0.02, shrink=0.6, aspect=40,
                       extend="both", spacing="uniform", ticks=breaks)
    bar.set_label(label, color=INK_SECONDARY, fontsize=9)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=8, length=0)
    bar.ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    priced = int(np.isfinite(values).sum())
    _titles(ax, title, f"{result.run} · LCOE year {result.priced_year} · {priced:,} priced pixels · "
                       f"grey: land without a result")
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)


def plot_distribution(result: Result, path: Path) -> None:
    values = result.lcom[np.isfinite(result.lcom)]
    low, high = np.quantile(values, [0.005, 0.995])
    median, p10 = float(np.median(values)), float(np.quantile(values, 0.1))
    fig = _figure(8.0, 4.2)
    ax = fig.add_subplot()
    _style(ax)
    ax.hist(values, bins=np.linspace(low, high, 61), color=SERIES, edgecolor=SURFACE, linewidth=0.6)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    top = ax.get_ylim()[1]
    for x, name in ((p10, "P10"), (median, "Median")):
        ax.axvline(x, color=INK_SECONDARY, linewidth=1)
        ax.text(x, top * 0.98, f" {name} {x:,.0f}", color=INK, fontsize=8.5, va="top", ha="left")
    ax.set_xlabel(f"LCOM, {result.currency}/t", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Pixels", color=INK_SECONDARY, fontsize=9)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:g}K" if v >= 1000 else f"{v:g}")
    _titles(ax, "Distribution of the LCOM over pixels",
            f"{result.run} · {len(values):,} priced pixels · 0.5th to 99.5th percentile shown · "
            "a pixel median weights a country by area")
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)


def plot_countries(result: Result, stats: pd.DataFrame, subtitle: str, path: Path) -> None:
    fig = _figure(8.0, 1.4 + 0.24 * len(stats))
    ax = fig.add_subplot()
    _style(ax)
    y = np.arange(len(stats))
    ax.hlines(y, stats["p10"], stats["p90"], color=SERIES_LIGHT, linewidth=2, capstyle="round")
    ax.scatter(stats["median"], y, s=36, color=SERIES, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    ax.set_yticks(y, [f"{iso3}  ({n:,})" for iso3, n in zip(stats.index, stats["n"], strict=True)])
    ax.set_ylim(-0.7, len(stats) - 0.3)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(f"LCOM, {result.currency}/t", color=INK_SECONDARY, fontsize=9)
    _titles(ax, "LCOM per country: median (dot) and P10 to P90 (line)", subtitle)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)


def plot_all(
    path: Path,
    out_dir: Path | None = None,
    currency: str = "EUR",
    breaks: list[float] | None = None,
    countries: list[str] | None = None,
    top: int = 30,
) -> list[Path]:
    """Write the four plots of one result file; returns their paths."""
    result = load(path, currency)
    if not np.isfinite(result.lcom).any():
        raise ValueError(f"{path.name} has no priced pixels.")
    if breaks is not None and (len(breaks) < 2 or np.any(np.diff(breaks) <= 0)):
        raise ValueError(f"breaks must be at least two rising values, got {breaks}.")
    out_dir = out_dir or path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{result.name}_{kind}.png" for kind in ("map_lcom", "map_electricity_share",
                                                                 "distribution", "countries")]
    share = result.electricity_share * 100
    stats = country_stats(result, countries, top)
    chosen = "selected countries" if countries else f"the {len(stats)} countries with the most priced pixels"
    with rc_context({"font.family": _font()}):
        plot_map(result, result.lcom, "Levelised cost of methanol (Route B)", f"LCOM, {currency}/t",
                 np.asarray(breaks, dtype=float) if breaks else class_breaks(result.lcom), paths[0])
        plot_map(result, share, "Electricity share of the LCOM", "Share of the LCOM that is electricity, %",
                 class_breaks(share), paths[1])
        plot_distribution(result, paths[2])
        plot_countries(result, stats, f"{result.run} · {chosen} · priced pixels in brackets", paths[3])
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lcom-plot", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", nargs="+", type=Path, help="LCOM result files (lcom_*.nc).")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output folder; default: next to each result.")
    parser.add_argument("--currency", choices=("EUR", "USD"), default="EUR", help="Cost unit (default: EUR).")
    parser.add_argument(
        "--breaks", type=float, nargs="+", metavar="VALUE", default=None,
        help="Class limits of the LCOM map, rising; default: the deciles of the priced pixels.",
    )
    parser.add_argument("--countries", type=lambda s: [c.strip().upper() for c in s.split(",") if c.strip()],
                        default=None, help="Comma-separated ISO3 codes for the country plot.")
    parser.add_argument("--top", type=int, default=30, help="Countries in the country plot (default: 30).")
    args = parser.parse_args(argv)
    try:
        for result in args.results:
            for written in plot_all(result, args.out_dir, args.currency, args.breaks, args.countries, args.top):
                print(written)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
