"""
Reviewed table of data flags: inputs whose source is unverified, differs from the source they
should come from, or rests on a simplification of this chain. Flags travel with the results: the
run logs a warning for each one, and the output file and assumptions register list them, so a
result cannot be separated from its caveats.

The cost-workbook flags come from the review of the BOA cost workbook. Each records the value the
workbook held when it was reviewed. ``check_cost_workbook_flags`` compares the workbook in use
against that value, so a flag does not go stale silently when the workbook changes.

Reference for the IRENA figures: IRENA, *Renewable Power Generation Costs in 2024*, July 2025,
Table 2.1 (onshore wind) and Table 3.1 (utility-scale solar PV), weighted-average total installed
cost in 2024 USD/kW. https://www.irena.org/Publications/2025/Jun/Renewable-Power-Generation-Costs-in-2024
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from lcom.paper_inputs import FLAG_DAC_PRODUCTIVITY_FIXED, FLAG_DAC_WEATHER_FIXED, FLAG_EUR_ANNUAL_RATE

logger = logging.getLogger(__name__)

STATUS_DEVIATES = "deviates-from-cited-source"
STATUS_STAND_IN = "country-stand-in-for-region"
STATUS_NOT_IDENTIFIED = "source-not-identified"
STATUS_EXCLUDED = "pixels-excluded"
STATUS_SIMPLIFIED = "simplification"

IRENA_2024 = "IRENA, Renewable Power Generation Costs in 2024 (2025), Table 2.1 / Table 3.1"
_BATTERY_NOTE = (
    "IRENA (2025) reports a global 2024 utility-scale battery storage system price of about 192 USD/kWh; "
    "the workbook value is not from that report."
)


@dataclass(frozen=True)
class DataFlag:
    key: str
    status: str
    description: str
    # Where the value sits in the cost workbook, for flags that can be checked against it.
    region: str | None = None
    tech: str | None = None  # "Solar PV" | "Onshore wind" | "Battery"
    workbook_value: float | None = None  # 2024 value when the flag was reviewed
    reference_value: float | None = None  # the value in the cited source, if it has one
    reference: str = ""


COST_WORKBOOK_FLAGS: tuple[DataFlag, ...] = (
    DataFlag(
        "capex.china.solar",
        STATUS_DEVIATES,
        "China solar PV CAPEX differs from IRENA (591 USD/kW); source of the workbook value not identified.",
        "China",
        "Solar PV",
        472.0,
        591.0,
        IRENA_2024,
    ),
    DataFlag(
        "capex.china.wind",
        STATUS_DEVIATES,
        "China onshore wind CAPEX differs from IRENA (856 USD/kW); source of the workbook value not identified.",
        "China",
        "Onshore wind",
        753.0,
        856.0,
        IRENA_2024,
    ),
    DataFlag(
        "capex.north_america.wind",
        STATUS_DEVIATES,
        "North America onshore wind CAPEX matches neither IRENA's United States (1568) nor Other North "
        "America (1238); it looks like a blend. Source not identified.",
        "North America",
        "Onshore wind",
        1486.4375,
        None,
        IRENA_2024,
    ),
    DataFlag(
        "capex.north_america.solar",
        STATUS_STAND_IN,
        "North America solar PV CAPEX is the IRENA United States value (1058); IRENA's Other North America is 1206.",
        "North America",
        "Solar PV",
        1058.0,
        1206.0,
        IRENA_2024,
    ),
    DataFlag(
        "capex.south_america.solar",
        STATUS_STAND_IN,
        "South America solar PV CAPEX is the IRENA Brazil value (724); IRENA's Other South America is 1048.",
        "South America",
        "Solar PV",
        724.0,
        1048.0,
        IRENA_2024,
    ),
    DataFlag(
        "capex.south_america.wind",
        STATUS_STAND_IN,
        "South America onshore wind CAPEX is the IRENA Brazil value (1307); IRENA's Other South America is 1492.",
        "South America",
        "Onshore wind",
        1307.0,
        1492.0,
        IRENA_2024,
    ),
    DataFlag(
        "capex.middle_east.solar",
        STATUS_NOT_IDENTIFIED,
        "Middle East solar PV CAPEX has no IRENA 2024 region of its own; source of 583.13 USD/kW not identified.",
        "Middle East",
        "Solar PV",
        583.126166,
        None,
        IRENA_2024,
    ),
    DataFlag(
        "capex.middle_east.wind",
        STATUS_STAND_IN,
        "Middle East onshore wind CAPEX equals IRENA's Eurasia value (1425).",
        "Middle East",
        "Onshore wind",
        1425.0,
        1425.0,
        IRENA_2024,
    ),
    DataFlag(
        "capex.battery.global",
        STATUS_DEVIATES,
        "Battery CAPEX (125 USD/kWh in every region except China) does not match IRENA. " + _BATTERY_NOTE,
        "India",
        "Battery",
        125.0,
        192.0,
        IRENA_2024,
    ),
    DataFlag(
        "capex.battery.china",
        STATUS_NOT_IDENTIFIED,
        "China battery CAPEX (84.36 USD/kWh) source not identified. " + _BATTERY_NOTE,
        "China",
        "Battery",
        84.36,
        192.0,
        IRENA_2024,
    ),
)

# BOA flags without a numeric check against the cost workbook. They always apply.
STANDING_FLAGS: tuple[DataFlag, ...] = (
    DataFlag(
        "opex.world",
        STATUS_NOT_IDENTIFIED,
        "RES OPEX fractions (solar 1 %, wind 2 %, battery 2 % of CAPEX per year) are not in the IRENA 2024 "
        "report text; source not identified.",
    ),
    DataFlag(
        "capex.trajectory",
        STATUS_NOT_IDENTIFIED,
        "The 2025-2050 CAPEX trajectories are the workbook's own projections; origin not identified.",
    ),
    DataFlag(
        "wacc.country",
        STATUS_NOT_IDENTIFIED,
        "Country cost of capital: the cost of equity is consistent with Damodaran's total equity risk premium "
        "by country (ctryprem, January 2026) plus a common risk-free rate and a beta of about 1.13, but the "
        "derivation, the cost of debt and the capital structure are not recorded. This chain uses the same "
        "WACC for every stage, so the flag applies to every cost it reports.",
    ),
    DataFlag(
        "domain.greenland",
        STATUS_EXCLUDED,
        "Greenland is not priced. The BOA regions leave Greenland between 50W and 25W and north of 72N "
        "outside the model, so only two coastal strips (about 16 % of its land south of 72N) have an LCOE.",
    ),
)

# Flags of this chain's inputs and simplifications.
CHAIN_FLAGS: tuple[DataFlag, ...] = (
    DataFlag(
        FLAG_EUR_ANNUAL_RATE,
        STATUS_SIMPLIFIED,
        "Paper costs are January 2024 EUR (HICP adjusted); they are converted with one rate, the ECB 2024 "
        "annual average (1.08238 USD/EUR). BOA costs are 2024 USD.",
    ),
    DataFlag(
        FLAG_DAC_WEATHER_FIXED,
        STATUS_SIMPLIFIED,
        "DAC electricity and heat demand and the heat pump COP are one value for every pixel: the medians of "
        "the paper's 20,265 regions. They depend on temperature and humidity; in hot dry regions (mean above "
        "25 C, relative humidity below 35 %) the median DAC electricity is 0.53 instead of 0.45 MWh/tCO2.",
    ),
    DataFlag(
        FLAG_DAC_PRODUCTIVITY_FIXED,
        STATUS_SIMPLIFIED,
        "DAC relative productivity is 1 for every pixel (paper median 1.005). In hot dry regions the paper's "
        "median is 0.65, which raises the DAC capital and O&M per tonne by a factor of 1.55 there.",
    ),
    DataFlag(
        "heat-integration-not-modelled",
        STATUS_SIMPLIFIED,
        "All DAC heat comes from an air-source heat pump. Waste heat of electrolysis and synthesis (about "
        "2.9 MWh/t_MeOH at 40 C) and purge gas heat (0.49 MWh/t_MeOH) are not used. Estimated effect: about "
        "0.5 MWh_el/t_MeOH, 3 to 4 % of LCOM, which this chain overstates.",
    ),
)

ALL_FLAGS = COST_WORKBOOK_FLAGS + STANDING_FLAGS + CHAIN_FLAGS


@dataclass(frozen=True)
class FlagResult:
    flag: DataFlag
    found: float | None
    unchanged: bool  # the workbook still holds the reviewed value


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {c: str(c).strip().lower().replace("_", " ") if not isinstance(c, int) else c for c in df.columns}
    df = df.rename(columns=renamed)
    return df.rename(columns={"technology": "tech"})


def check_cost_workbook_flags(path: Path, year: int = 2024) -> list[FlagResult]:
    """
    Compare each cost-workbook flag with the workbook at ``path``. A flag whose workbook value
    has changed is reported with ``unchanged=False``: its review no longer holds.
    """
    capex = _normalise_columns(pd.read_excel(path, sheet_name="RES CAPEX projections"))
    capex["irena region"] = capex["irena region"].astype(str).str.strip()
    if year not in capex.columns:
        raise ValueError(f"RES CAPEX projections in {path.name} has no {year} column.")
    results = []
    for flag in COST_WORKBOOK_FLAGS:
        rows = capex[(capex["irena region"] == flag.region) & (capex["tech"] == flag.tech)]
        found = float(rows[year].iloc[0]) if len(rows) == 1 else None
        unchanged = (
            found is not None
            and flag.workbook_value is not None
            and abs(found - flag.workbook_value) <= 1e-4 * abs(flag.workbook_value)
        )
        results.append(FlagResult(flag, found, unchanged))
    return results


def log_flags(results: list[FlagResult]) -> None:
    """One warning per flag, so a run log states every caveat that applies."""
    for result in results:
        flag = result.flag
        if not result.unchanged:
            logger.warning(
                "[DATA FLAG STALE] %s: the cost workbook value is now %s, not the reviewed %s; review the flag.",
                flag.key,
                result.found,
                flag.workbook_value,
            )
        logger.warning("[DATA FLAG] %s (%s): %s", flag.key, flag.status, flag.description)
    for flag in (*STANDING_FLAGS, *CHAIN_FLAGS):
        logger.warning("[DATA FLAG] %s (%s): %s", flag.key, flag.status, flag.description)
