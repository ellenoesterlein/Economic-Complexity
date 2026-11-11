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
- ``<stem>_breakdown_top5.png`` and ``.csv``: the LCOM of the cheapest 5 % of each country's pixels
  (``--share``), split into cost items: the electricity of all stages from solar, wind and battery, and the
  electrolyser, DAC, heat pump and synthesis plant. Each bar is the mean over these pixels, so the items add up
  to the mean LCOM. Whiskers show P10 to P90 of each electricity part. The plot has three groups: the cheapest
  10 countries (``--cheapest``), the countries of interest (``--interest``) and the 5 most expensive
  (``--expensive``); the first and last only from countries with at least 20 priced pixels. The CSV has every
  country. Only for result files with cost items (``lcom_item_*``).

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
# Categorical slots 1 to 7 of the reference palette, in their fixed order: the stack order of the breakdown.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
# (result variable item, label); the three electricity parts first, then the plants along the chain.
BREAKDOWN_ITEMS = (
    ("electricity_solar", "Electricity: solar"),
    ("electricity_wind", "Electricity: wind"),
    ("electricity_battery", "Electricity: battery"),
    ("electrolyser", "Electrolyser"),
    ("dac", "DAC plant"),
    ("heat_pump", "Heat pump"),
    ("synthesis", "Synthesis plant"),
)
ELECTRICITY_ITEMS = ("electricity_solar", "electricity_wind", "electricity_battery")
TOP_SHARE = 0.05
CHEAPEST = 10
MOST_EXPENSIVE = 5
# Countries of interest: today's methanol producers TTO, CHN, IRN, SAU, OMN, USA, RUS; CHL (planned e-fuel
# exports); DEU, JPN (importers). Change with --interest.
INTEREST = ("TTO", "CHN", "IRN", "SAU", "OMN", "USA", "RUS", "CHL", "DEU", "JPN")
# A country enters the cheapest or most expensive group only with at least this many priced pixels: below it,
# the cheapest 5 % is a single pixel.
MIN_PIXELS = 20


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


def load_items(path: Path, currency: str = "EUR") -> pd.DataFrame:
    """
    One row per priced pixel: ``iso3``, ``lcom`` and one column per breakdown item, per tonne in ``currency``.
    Raises ValueError for a result file without cost items.
    """
    rate = 1 / USD_PER_EUR if currency == "EUR" else 1.0
    with xr.open_dataset(path) as ds:
        names = [f"lcom_item_{item}_usd_per_t" for item, _ in BREAKDOWN_ITEMS]
        missing = [name for name in names if name not in ds.variables]
        if missing:
            raise ValueError(f"{path.name} has no cost items ({missing[0]}); price it again with lcom.cli.")
        legend = np.array([key.split(":")[0] for key in str(ds.attrs["cost_key_legend"]).split(",")])
        iso3 = legend[ds["cost_key_id"].values]
        layers = {"lcom": ds["lcom_usd_per_t"].isel(year=0).values.astype(float) * rate}
        for (item, _), name in zip(BREAKDOWN_ITEMS, names, strict=True):
            layers[item] = ds[name].isel(year=0).values.astype(float) * rate
    priced = np.isfinite(layers["lcom"]) & (iso3 != "") & (iso3 != "UNKNOWN")
    return pd.DataFrame({"iso3": iso3[priced], **{name: layer[priced] for name, layer in layers.items()}})


def breakdown_stats(
    items: pd.DataFrame, countries: list[str] | None = None, share: float = TOP_SHARE
) -> pd.DataFrame:
    """
    Per country (default: every country in ``items``), over the cheapest ``share`` of its pixels (at least one):
    pixel count, mean LCOM and the mean, P10 and P90 of every item. Sorted by mean LCOM.
    """
    if not 0 < share <= 1:
        raise ValueError(f"share must be in (0, 1], got {share}.")
    rows = []
    for iso3 in countries if countries is not None else sorted(items["iso3"].unique()):
        pixels = items[items["iso3"] == iso3]
        if pixels.empty:
            raise ValueError(f"No priced pixels for {iso3}.")
        top = pixels.nsmallest(int(np.ceil(share * len(pixels))), "lcom")
        row = {"iso3": iso3, "pixels": len(pixels), "n": len(top), "lcom_mean": top["lcom"].mean()}
        for item, _ in BREAKDOWN_ITEMS:
            row[f"{item}_mean"] = top[item].mean()
            row[f"{item}_p10"] = top[item].quantile(0.1)
            row[f"{item}_p90"] = top[item].quantile(0.9)
        rows.append(row)
    return pd.DataFrame(rows).set_index("iso3").sort_values("lcom_mean")


def breakdown_groups(
    stats: pd.DataFrame,
    cheapest: int = CHEAPEST,
    interest: tuple[str, ...] | list[str] = INTEREST,
    expensive: int = MOST_EXPENSIVE,
    min_pixels: int = MIN_PIXELS,
) -> list[tuple[str, pd.DataFrame]]:
    """
    The groups of the breakdown plot, each sorted by mean LCOM: the ``cheapest`` and the ``expensive`` most
    expensive countries with at least ``min_pixels`` priced pixels, and the countries of ``interest``. A country
    can be in more than one group. Countries of interest without priced pixels and empty groups are left out.
    """
    missing = sorted(set(interest) - set(stats.index))
    if missing:
        print(f"warning: no priced pixels for {', '.join(missing)}; left out of the breakdown plot.",
              file=sys.stderr)
        interest = [iso3 for iso3 in interest if iso3 in stats.index]
    eligible = stats[stats["pixels"] >= min_pixels]
    groups = [
        (f"Cheapest {cheapest}", eligible.head(cheapest) if cheapest > 0 else eligible.iloc[:0]),
        ("Countries of interest", stats.loc[list(interest)].sort_values("lcom_mean")),
        (f"Most expensive {expensive}", eligible.tail(expensive) if expensive > 0 else eligible.iloc[:0]),
    ]
    return [(title, group) for title, group in groups if not group.empty]


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
    lines = subtitle.count("\n") + 1
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold", pad=8 + 12 * lines)
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


def plot_breakdown(groups: list[tuple[str, pd.DataFrame]], currency: str, subtitle: str, path: Path) -> None:
    """
    Stacked bars of the item means, one block of rows per group under its title; whiskers at the end of each
    electricity segment show its P10 to P90.
    """
    stats = pd.concat([group for _, group in groups])
    # Row positions: each group starts one row lower than the end of the last one; its title sits in that gap.
    y, titles, row = [], [], 0.0
    for title, group in groups:
        titles.append((title, row))
        y.extend(row + 1 + np.arange(len(group)))
        row = y[-1] + 1
    y = np.asarray(y, dtype=float)
    fig = _figure(9.0, 1.9 + 0.3 * (len(stats) + len(groups)))
    ax = fig.add_subplot()
    _style(ax)
    left = np.zeros(len(stats))
    # The three whiskers sit at different heights inside the bar, so that they do not overlap.
    whisker_offset = dict(zip(ELECTRICITY_ITEMS, (-0.2, 0.0, 0.2), strict=True))
    for (item, label), color in zip(BREAKDOWN_ITEMS, CATEGORICAL, strict=True):
        mean = stats[f"{item}_mean"].to_numpy()
        ax.barh(y, mean, left=left, height=0.66, color=color, edgecolor=SURFACE, linewidth=1.0, label=label)
        if item in whisker_offset:
            end = left + mean
            low = end - (mean - stats[f"{item}_p10"].to_numpy())
            high = end + (stats[f"{item}_p90"].to_numpy() - mean)
            yy = y + whisker_offset[item]
            ax.hlines(yy, low, high, color=INK, linewidth=0.9, zorder=3)
            ax.vlines(low, yy - 0.06, yy + 0.06, color=INK, linewidth=0.9, zorder=3)
            ax.vlines(high, yy - 0.06, yy + 0.06, color=INK, linewidth=0.9, zorder=3)
        left = left + mean
    for yi, total in zip(y, stats["lcom_mean"], strict=True):
        ax.text(total, yi, f"  {total:,.0f}", va="center", ha="left", fontsize=7.5, color=INK_SECONDARY)
    labels = [f"{iso3}  ({n:,}/{total:,})" for iso3, n, total in zip(stats.index, stats["n"], stats["pixels"],
                                                                      strict=True)]
    ax.set_yticks(y, labels)
    for title, at in titles:
        ax.text(0, at + 0.15, title, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=9,
                fontweight="bold", color=INK)
    ax.set_ylim(-0.7, y[-1] + 0.7)
    ax.invert_yaxis()
    ax.set_xlim(0, float(left.max()) * 1.08)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.set_xlabel(f"Mean LCOM of the cheapest pixels, {currency}/t", color=INK_SECONDARY, fontsize=9)
    legend = fig.legend(loc="outside lower center", ncol=4, frameon=False, fontsize=8, handlelength=1.2,
                        handleheight=1.0, columnspacing=1.4)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    _titles(ax, "LCOM cost items of the cheapest pixels per country", subtitle)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)


def write_breakdown(stats: pd.DataFrame, currency: str, path: Path) -> None:
    unit = currency.lower()
    stats.rename(columns=lambda c: c if c in ("pixels", "n") else f"{c}_{unit}_per_t").round(1).to_csv(path)


def plot_all(
    path: Path,
    out_dir: Path | None = None,
    currency: str = "EUR",
    breaks: list[float] | None = None,
    countries: list[str] | None = None,
    top: int = 30,
    share: float = TOP_SHARE,
    cheapest: int = CHEAPEST,
    interest: tuple[str, ...] | list[str] = INTEREST,
    expensive: int = MOST_EXPENSIVE,
) -> list[Path]:
    """Write the plots of one result file (and the breakdown table, if it has cost items); returns their paths."""
    result = load(path, currency)
    if not np.isfinite(result.lcom).any():
        raise ValueError(f"{path.name} has no priced pixels.")
    if breaks is not None and (len(breaks) < 2 or np.any(np.diff(breaks) <= 0)):
        raise ValueError(f"breaks must be at least two rising values, got {breaks}.")
    out_dir = out_dir or path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{result.name}_{kind}.png" for kind in ("map_lcom", "map_electricity_share",
                                                                 "distribution", "countries")]
    electricity_pct = result.electricity_share * 100
    stats = country_stats(result, countries, top)
    chosen = "selected countries" if countries else f"the {len(stats)} countries with the most priced pixels"
    with rc_context({"font.family": _font()}):
        plot_map(result, result.lcom, "Levelised cost of methanol (Route B)", f"LCOM, {currency}/t",
                 np.asarray(breaks, dtype=float) if breaks else class_breaks(result.lcom), paths[0])
        plot_map(result, electricity_pct, "Electricity share of the LCOM", "Share of the LCOM that is electricity, %",
                 class_breaks(electricity_pct), paths[1])
        plot_distribution(result, paths[2])
        plot_countries(result, stats, f"{result.run} · {chosen} · priced pixels in brackets", paths[3])
        try:
            items = load_items(path, currency)
        except ValueError as e:
            print(f"warning: no breakdown plot: {e}", file=sys.stderr)
            return paths
        breakdown = breakdown_stats(items, share=share)
        stem = f"{result.name}_breakdown_top{share * 100:g}"
        plot_breakdown(
            breakdown_groups(breakdown, cheapest, interest, expensive),
            currency,
            f"{result.run} · cheapest {share:.0%} of each country's pixels (used/priced in brackets) · bars: means\n"
            f"whiskers: P10 to P90 of each electricity part · cheapest and most expensive of the countries with "
            f"{MIN_PIXELS}+ pixels",
            out_dir / f"{stem}.png",
        )
        write_breakdown(breakdown, currency, out_dir / f"{stem}.csv")
        paths += [out_dir / f"{stem}.png", out_dir / f"{stem}.csv"]
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
    parser.add_argument(
        "--share", type=float, default=TOP_SHARE,
        help="Share of each country's cheapest pixels in the breakdown plot (default: 0.05).",
    )
    parser.add_argument("--cheapest", type=int, default=CHEAPEST,
                        help=f"Cheapest countries in the breakdown plot (default: {CHEAPEST}).")
    parser.add_argument("--expensive", type=int, default=MOST_EXPENSIVE,
                        help=f"Most expensive countries in the breakdown plot (default: {MOST_EXPENSIVE}).")
    parser.add_argument(
        "--interest", type=lambda s: [c.strip().upper() for c in s.split(",") if c.strip()], default=list(INTEREST),
        help=f"Comma-separated ISO3 codes of the countries of interest (default: {','.join(INTEREST)}).",
    )
    args = parser.parse_args(argv)
    try:
        for result in args.results:
            for written in plot_all(
                result, args.out_dir, args.currency, args.breaks, args.countries, args.top, args.share,
                args.cheapest, args.interest, args.expensive,
            ):
                print(written)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
