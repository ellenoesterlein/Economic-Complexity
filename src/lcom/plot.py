"""
Plot an LCOM result: world maps, the distribution over pixels, the country medians and the cost breakdown.

For each result file (``lcom_*.nc``) it writes to ``figures/`` and ``tables/`` next to it (with ``--out-dir DIR``:
to ``DIR/<result stem>/figures`` and ``/tables``). The maps use the Equal Earth projection (cartopy).

- ``map_lcom_classes.png``: LCOM per pixel in 8 classes from blue (cheaper than the median) to red (more expensive).
  Class limits: percentiles 5, 15, 30, 50, 70, 85 and 95 of the priced pixels, or ``--breaks``; give the same
  breaks to compare runs.
- ``map_lcom_continuous.png``: LCOM per pixel on the continuous batlow scale (Crameri's Scientific Colour Maps;
  cheap is dark blue), from percentile 5 to 95 of the priced pixels.
- ``map_electricity_share.png``: the share of the LCOM that is electricity (electrolysis, DAC, heat pump and
  synthesis plant), in decile classes.
- ``histogram_lcom_pixels.png``: histogram of the LCOM over the priced pixels, with the median and the P10.
- ``countries_lcom_median_p10_p90.png``: median and P10 to P90 over all priced pixels of each country.
- ``countries_cost_items_cheapest5pct.png`` and ``tables/countries_cost_items_cheapest5pct.csv``: the LCOM of
  the cheapest 5 % of each country's pixels (``--share``), split into cost items: the electricity of all stages
  from solar, wind and battery, and the electrolyser, DAC, heat pump and synthesis plant. Each bar is the mean
  over these pixels, so the items add up to the mean LCOM. Whiskers show P10 to P90 of each electricity part.
  The CSV has every country. Only for result files with cost items (``lcom_item_*``).

The two country plots show the same three groups: the cheapest 10 countries (``--cheapest``), the countries of
interest (``--interest``) and the 5 most expensive (``--expensive``), ranked by the mean LCOM of the cheapest 5 %
of each country's pixels (by the median, for a result without cost items). The first and last group are only
from countries with at least 20 priced pixels. ``--countries`` gives one group of selected countries instead.

Borders are the edges between the cost keys of the result file (country level, 0.25 degree steps), so no
shapefile is necessary. Grey land has a cost key but no result.

Examples:
    python -m lcom.plot outputs/<run>/*/lcom_*.nc
    python -m lcom.plot RESULT.nc --currency USD --breaks 500 600 700 800 900 1000 1200 1500
    python -m lcom.plot RESULT.nc --countries CHN,IND,SAU,AUS,USA,ARG,MEX,ZAF,FRA
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cartopy.crs as ccrs
import colormaps as cmaps
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import font_manager, rc_context
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize, to_hex
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
# Diverging ramp of the LCOM map: the blue sequential steps 700, 550, 350, 150, the neutral grey, and red steps
# (hue of categorical red) with the same OKLCH lightness as the blue steps.
DIVERGING = ["#0d366b", "#1c5cab", "#5598e7", "#b7d3f6", "#f0efec", "#f4c2be", "#dd716a", "#9e3432", "#621b1a"]
MAP_PERCENTILES = (5, 15, 30, 50, 70, 85, 95)
# Continuous LCOM map: batlow of Crameri's Scientific Colour Maps (perceptually uniform, readable with colour-vision
# deficiency and in grey); cheap is dark blue. Its range is percentiles 5 to 95, like the end classes of map_lcom.
CONTINUOUS = "batlow"
CONTINUOUS_RANGE = (5, 95)
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
# File names of the figures of every result, in figures/; the cost-item figure and table follow when the result
# has cost items.
FIGURES = ("map_lcom_classes", "map_lcom_continuous", "map_electricity_share", "histogram_lcom_pixels",
           "countries_lcom_median_p10_p90")


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


def country_groups(
    stats: pd.DataFrame,
    cheapest: int = CHEAPEST,
    interest: tuple[str, ...] | list[str] = INTEREST,
    expensive: int = MOST_EXPENSIVE,
    min_pixels: int = MIN_PIXELS,
    by: str = "lcom_mean",
) -> list[tuple[str, pd.DataFrame]]:
    """
    The country groups of the country plots, each sorted by column ``by``: the ``cheapest`` and the ``expensive``
    most expensive countries with at least ``min_pixels`` priced pixels, and the countries of ``interest``. A
    country can be in more than one group. Countries of interest without priced pixels and empty groups are left
    out.
    """
    stats = stats.sort_values(by)
    missing = sorted(set(interest) - set(stats.index))
    if missing:
        print(f"warning: no priced pixels for {', '.join(missing)}; left out of the country plots.",
              file=sys.stderr)
        interest = [iso3 for iso3 in interest if iso3 in stats.index]
    eligible = stats[stats["pixels"] >= min_pixels]
    groups = [
        (f"Cheapest {cheapest}", eligible.head(cheapest) if cheapest > 0 else eligible.iloc[:0]),
        ("Countries of interest", stats.loc[list(interest)].sort_values(by)),
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


def country_stats(result: Result, countries: list[str] | None = None) -> pd.DataFrame:
    """Priced pixel count, median, P10 and P90 per country (default: every country), sorted by median."""
    priced = np.isfinite(result.lcom) & (result.iso3 != "")
    df = pd.DataFrame({"iso3": result.iso3[priced], "lcom": result.lcom[priced]})
    stats = df.groupby("iso3")["lcom"].agg(
        pixels="size", median="median", p10=lambda v: v.quantile(0.1), p90=lambda v: v.quantile(0.9)
    )
    if countries:
        missing = sorted(set(countries) - set(stats.index))
        if missing:
            raise ValueError(f"No priced pixels for {', '.join(missing)}.")
        stats = stats.loc[countries]
    return stats.sort_values("median")


def _group_rows(groups: list[tuple[str, pd.DataFrame]]) -> tuple[pd.DataFrame, np.ndarray, list[tuple[str, float]]]:
    """
    The rows of all groups in one frame, their y positions and the y position of each group title: each group
    starts one row lower than the end of the last one, and its title sits in that gap.
    """
    y, titles, row = [], [], 0.0
    for title, group in groups:
        titles.append((title, row))
        y.extend(row + 1 + np.arange(len(group)))
        row = y[-1] + 1
    return pd.concat([group for _, group in groups]), np.asarray(y, dtype=float), titles


def _group_axis(ax, y: np.ndarray, labels: list[str], titles: list[tuple[str, float]]) -> None:
    """Country labels, bold group titles and the y range of a grouped country plot (top row first)."""
    ax.set_yticks(y, labels)
    for title, at in titles:
        ax.text(0, at + 0.15, title, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=9,
                fontweight="bold", color=INK)
    ax.set_ylim(-0.7, y[-1] + 0.7)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")


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


def percentile_breaks(values: np.ndarray, percentiles: tuple[float, ...] = MAP_PERCENTILES) -> np.ndarray:
    """Class limits at ``percentiles`` of ``values``, rounded to a twentieth of the order of magnitude of their range."""
    limits = np.percentile(values[np.isfinite(values)], percentiles)
    spread = float(np.ptp(limits)) or 1.0
    step = 10 ** np.floor(np.log10(spread)) / 20
    return np.unique(np.round(limits / step) * step)


def diverging_colors(classes: int) -> list[str]:
    """``classes`` colours from dark blue (low) to dark red (high); the neutral grey only for an odd count."""
    ramp = LinearSegmentedColormap.from_list("diverging", DIVERGING)
    arm = classes // 2
    low = np.linspace(0, 0.5, arm + 1)[:-1]
    positions = [*low, *([0.5] if classes % 2 else []), *(1 - low[::-1])]
    return [to_hex(ramp(p)) for p in positions]


def plot_map(
    result: Result,
    values: np.ndarray,
    title: str,
    label: str,
    breaks: np.ndarray,
    path: Path,
    diverging: bool = False,
    note: str = "",
    continuous: bool = False,
) -> None:
    """
    A map in the Equal Earth projection. Classed: one colour per class between ``breaks``, plus one class below
    and one above; ``diverging`` colours low classes blue and high classes red, otherwise one blue ramp.
    ``continuous``: the batlow scale from the first to the last value of ``breaks``, extended at both ends.
    """
    land = result.iso3 != ""
    rows = np.nonzero(land.any(axis=1))[0]
    lat_range = (float(result.lat[rows].min()) - 1, float(result.lat[rows].max()) + 1)
    # The grid runs from -180 to 180 with both ends; drop the repeated last column. Keep the rows of the map.
    window = (slice(rows.min(), rows.max() + 1), slice(0, len(result.lon) - 1))
    lat, lon = result.lat[window[0]], result.lon[window[1]]
    dlat, dlon = abs(float(lat[1] - lat[0])), abs(float(lon[1] - lon[0]))
    extent = (lon.min() - dlon / 2, lon.max() + dlon / 2, lat.min() - dlat / 2, lat.max() + dlat / 2)
    origin = "lower" if lat[0] < lat[-1] else "upper"
    layer = np.ma.masked_invalid(values[window].astype(float))
    without = np.ma.masked_invalid(np.where(land[window] & ~np.isfinite(values[window]), 1.0, np.nan))

    projection, lonlat = ccrs.EqualEarth(), ccrs.PlateCarree()
    corners = projection.transform_points(lonlat, np.array([-180.0, 180.0, 0.0, 0.0]),
                                          np.array([0.0, 0.0, *lat_range]))
    width = 12.0
    aspect = (corners[3, 1] - corners[2, 1]) / (corners[1, 0] - corners[0, 0])
    fig = _figure(width, width * aspect + 1.9)
    ax = fig.add_subplot(projection=projection)
    ax.set_facecolor(SURFACE)
    # Limits in projected units: the full width at the equator, the height of the land rows. set_extent would take
    # the width at the cropped latitudes and cut the map at the sides.
    margin = 0.005 * (corners[1, 0] - corners[0, 0])
    ax.set_xlim(corners[0, 0] - margin, corners[1, 0] + margin)
    ax.set_ylim(corners[2, 1], corners[3, 1])
    ax.spines["geo"].set_edgecolor(AXIS)
    ax.spines["geo"].set_linewidth(0.6)
    ax.gridlines(crs=lonlat, xlocs=range(-180, 181, 30), ylocs=range(-60, 61, 30), color=GRID, linewidth=0.4,
                 zorder=0)

    # Nearest-neighbour regridding keeps every class colour intact. Target: four cells per grid cell along the long
    # side, at most 1800 along the short side (the 0.25 degree world grid reaches the cap).
    regrid = {"transform": lonlat, "extent": extent, "origin": origin, "interpolation": "nearest",
              "regrid_shape": min(1800, 4 * max(layer.shape))}
    ax.imshow(without, cmap=ListedColormap([NO_RESULT]), zorder=1, **regrid)
    if continuous:
        cmap = getattr(cmaps, CONTINUOUS).with_extremes(bad=(0, 0, 0, 0))
        norm = Normalize(float(breaks[0]), float(breaks[-1]))
    else:
        if diverging:
            colors = diverging_colors(len(breaks) + 1)
        else:
            colors = [to_hex(c) for c in LinearSegmentedColormap.from_list("blue", BLUE_RAMP)(
                np.linspace(0, 1, len(breaks) + 1))]
        cmap = ListedColormap(colors[1:-1]).with_extremes(under=colors[0], over=colors[-1], bad=(0, 0, 0, 0))
        norm = BoundaryNorm(breaks, len(breaks) - 1)
    shown = ax.imshow(layer, cmap=cmap, norm=norm, zorder=2, **regrid)
    # Border segments are 0.25 degree long, so projecting their end points is exact enough.
    segments = border_segments(result.iso3, result.lat, result.lon)
    ends = projection.transform_points(lonlat, segments[..., 0].ravel(), segments[..., 1].ravel())[:, :2]
    ax.add_collection(LineCollection(ends.reshape(-1, 2, 2), colors=BORDER, linewidths=0.3, zorder=3))

    bar = fig.colorbar(shown, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03, shrink=0.62, aspect=42,
                       extend="both", extendfrac=0.06, spacing="uniform", ticks=None if continuous else breaks)
    bar.set_label(label + (f"\n{note}" if note else ""), color=INK_SECONDARY, fontsize=9)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=INK_MUTED, labelcolor=INK_SECONDARY, labelsize=8.5, length=0, pad=4)
    bar.ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    if diverging or continuous:
        for x, text, align in ((0, "← cheaper", "left"), (1, "more expensive →", "right")):
            bar.ax.text(x, 1.5, text, transform=bar.ax.transAxes, ha=align, va="bottom", fontsize=8.5,
                        color=INK_SECONDARY)
    priced = int(np.isfinite(values).sum())
    _titles(ax, title, f"{result.run} · LCOE year {result.priced_year} · {priced:,} priced pixels · "
                       f"Equal Earth projection · grey: land without a result")
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


def plot_countries(groups: list[tuple[str, pd.DataFrame]], currency: str, subtitle: str, path: Path) -> None:
    """Median (dot) and P10 to P90 (line) over all priced pixels of each country, one block of rows per group."""
    stats, y, titles = _group_rows(groups)
    fig = _figure(8.0, 1.6 + 0.26 * (len(stats) + len(groups)))
    ax = fig.add_subplot()
    _style(ax)
    ax.hlines(y, stats["p10"], stats["p90"], color=SERIES_LIGHT, linewidth=2, capstyle="round")
    ax.scatter(stats["median"], y, s=36, color=SERIES, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    _group_axis(ax, y, [f"{iso3}  ({n:,})" for iso3, n in zip(stats.index, stats["pixels"], strict=True)], titles)
    ax.set_xlabel(f"LCOM, {currency}/t", color=INK_SECONDARY, fontsize=9)
    _titles(ax, "LCOM per country: median (dot) and P10 to P90 (line) of all priced pixels", subtitle)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)


def plot_breakdown(groups: list[tuple[str, pd.DataFrame]], currency: str, subtitle: str, path: Path) -> None:
    """
    Stacked bars of the item means, one block of rows per group under its title; whiskers at the end of each
    electricity segment show its P10 to P90.
    """
    stats, y, titles = _group_rows(groups)
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
    _group_axis(ax, y, labels, titles)
    ax.set_xlim(0, float(left.max()) * 1.08)
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
    share: float = TOP_SHARE,
    cheapest: int = CHEAPEST,
    interest: tuple[str, ...] | list[str] = INTEREST,
    expensive: int = MOST_EXPENSIVE,
) -> list[Path]:
    """
    Write the plots of one result file (and the cost-item table, if it has cost items); returns their paths. Both
    country plots show the same countries: ``countries``, or the groups of ``country_groups``, ranked by the mean
    LCOM of the cheapest ``share`` of each country's pixels (by the median, for a result without cost items).
    """
    result = load(path, currency)
    if not np.isfinite(result.lcom).any():
        raise ValueError(f"{path.name} has no priced pixels.")
    if breaks is not None and (len(breaks) < 2 or np.any(np.diff(breaks) <= 0)):
        raise ValueError(f"breaks must be at least two rising values, got {breaks}.")
    base = out_dir / result.name if out_dir else path.parent
    figures, tables = base / "figures", base / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    paths = [figures / f"{name}.png" for name in FIGURES]
    electricity_pct = result.electricity_share * 100
    stats = country_stats(result, countries)
    try:
        breakdown = breakdown_stats(load_items(path, currency), share=share)
    except ValueError as e:
        print(f"warning: no cost-item plot: {e}", file=sys.stderr)
        breakdown = None
    if countries:
        groups = [("Selected countries", stats)]
        chosen = "selected countries"
    elif breakdown is not None:
        groups = country_groups(breakdown, cheapest, interest, expensive)
        chosen = (f"groups as in the cost-item plot: ranked by the mean LCOM of each country's cheapest {share:.0%}\n"
                  f"cheapest and most expensive only of the countries with {MIN_PIXELS}+ priced pixels")
    else:
        groups = country_groups(stats, cheapest, interest, expensive, by="median")
        chosen = (f"groups ranked by median\ncheapest and most expensive only of the countries with {MIN_PIXELS}+ "
                  "priced pixels")
    with rc_context({"font.family": _font()}):
        plot_map(
            result,
            result.lcom,
            "Levelised cost of methanol (Route B)",
            f"LCOM, {currency}/t",
            np.asarray(breaks, dtype=float) if breaks else percentile_breaks(result.lcom),
            paths[0],
            diverging=True,
            note="class limits: --breaks" if breaks else
            "class limits: percentiles 5, 15, 30, 50 (median), 70, 85 and 95 of the priced pixels",
        )
        plot_map(
            result,
            result.lcom,
            "Levelised cost of methanol (Route B)",
            f"LCOM, {currency}/t",
            np.percentile(result.lcom[np.isfinite(result.lcom)], CONTINUOUS_RANGE),
            paths[1],
            note=f"continuous scale (batlow, Crameri) from percentile {CONTINUOUS_RANGE[0]} to {CONTINUOUS_RANGE[1]} of the "
            "priced pixels",
            continuous=True,
        )
        plot_map(result, electricity_pct, "Electricity share of the LCOM", "Share of the LCOM that is electricity, %",
                 class_breaks(electricity_pct), paths[2])
        plot_distribution(result, paths[3])
        plot_countries(
            [(title, stats.loc[group.index].sort_values("median")) for title, group in groups],
            currency,
            f"{result.run} · priced pixels in brackets · sorted by median in each group\n{chosen}",
            paths[4],
        )
        if breakdown is None:
            return paths
        stem = f"countries_cost_items_cheapest{share * 100:g}pct"
        tables.mkdir(parents=True, exist_ok=True)
        plot_breakdown(
            [(title, breakdown.loc[group.index]) for title, group in groups],
            currency,
            f"{result.run} · cheapest {share:.0%} of each country's pixels (used/priced in brackets) · bars: means\n"
            "whiskers: P10 to P90 of each electricity part · "
            + ("selected countries" if countries else
               f"cheapest and most expensive of the countries with {MIN_PIXELS}+ pixels"),
            figures / f"{stem}.png",
        )
        write_breakdown(breakdown, currency, tables / f"{stem}.csv")
        paths += [figures / f"{stem}.png", tables / f"{stem}.csv"]
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lcom-plot", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", nargs="+", type=Path, help="LCOM result files (lcom_*.nc).")
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output folder; files go to DIR/<result stem>/figures and /tables. Default: next to each result.",
    )
    parser.add_argument("--currency", choices=("EUR", "USD"), default="EUR", help="Cost unit (default: EUR).")
    parser.add_argument(
        "--breaks", type=float, nargs="+", metavar="VALUE", default=None,
        help="Class limits of the classed LCOM map, rising; default: percentiles 5 to 95 of the priced pixels.",
    )
    parser.add_argument("--countries", type=lambda s: [c.strip().upper() for c in s.split(",") if c.strip()],
                        default=None,
                        help="Comma-separated ISO3 codes for the country plots, in place of the three groups.")
    parser.add_argument(
        "--share", type=float, default=TOP_SHARE,
        help="Share of each country's cheapest pixels in the cost items and the ranking (default: 0.05).",
    )
    parser.add_argument("--cheapest", type=int, default=CHEAPEST,
                        help=f"Cheapest countries in the country plots (default: {CHEAPEST}).")
    parser.add_argument("--expensive", type=int, default=MOST_EXPENSIVE,
                        help=f"Most expensive countries in the country plots (default: {MOST_EXPENSIVE}).")
    parser.add_argument(
        "--interest", type=lambda s: [c.strip().upper() for c in s.split(",") if c.strip()], default=list(INTEREST),
        help=f"Comma-separated ISO3 codes of the countries of interest (default: {','.join(INTEREST)}).",
    )
    args = parser.parse_args(argv)
    try:
        for result in args.results:
            for written in plot_all(
                result, args.out_dir, args.currency, args.breaks, args.countries, args.share,
                args.cheapest, args.interest, args.expensive,
            ):
                print(written)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
