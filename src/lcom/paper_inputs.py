"""
Route B process inputs from Wenzel et al. (2026), converted to the units of the cost chain.

Route B is PEM electrolysis, direct CO2 hydrogenation to methanol, and solid sorbent direct air
capture (DAC) with an air-source heat pump for the desorption heat. All values are for 2050. Costs
are January 2024 EUR (HICP adjusted) in the paper.

``PAPER_VALUES`` holds every value as the paper or its data states it, with its source.
``chain_inputs()`` converts them to the units of the chain:

- EUR to USD at the ECB 2024 annual average reference rate (``USD_PER_EUR``).
- Fixed O&M from a share of CAPEX to an absolute annual cost per unit of capacity.
- Methanol synthesis from per-mass and molar units to per-kWh_MeOH units. CAPEX per (t_MeOH/a)
  becomes CAPEX per kW_MeOH of nameplate capacity, which runs 8760 h/a.

Version 1 uses one value for the weather-dependent DAC inputs: the medians over the 20,265 regions
of the paper's figshare data. The DAC relative productivity is set to 1.
"""

from dataclasses import dataclass

from lcom.constants import (
    HOURS_IN_YEAR,
    KWH_PER_KG_H2_LHV,
    KWH_PER_KG_METHANOL_LHV,
    KWH_PER_MWH,
    MOLAR_MASS_CO2,
    MOLAR_MASS_H2,
    MOLAR_MASS_METHANOL,
)

PAPER = "Wenzel et al., Nat. Commun. 17, 8495 (2026), doi:10.1038/s41467-026-76865-x"
FIGSHARE = "doi:10.6084/m9.figshare.30664253, mergedAveragedInputs_Year2018.csv"

# ECB reference rate, annual average 2024.
USD_PER_EUR = 1.08238046875
ECB_SERIES = "EXR.A.USD.EUR.SP00.A"
ECB_ACCESSED = "2026-10-28"
ECB_SOURCE = f"ECB data portal, series {ECB_SERIES}, 2024 annual average, accessed {ECB_ACCESSED}"

STAGES = ("electrolysis", "methanol", "dac", "heat_pump")

# How a value was obtained.
STATUS_TABLE = "paper-table"  # Supplementary Table 2
STATUS_TEXT = "paper-text"  # main text, Methods
STATUS_FIGSHARE = "figshare-median"  # median over the figshare regions
STATUS_CHOICE = "model-choice"  # set by this chain, not by the paper

# Data flags. lcom.flags describes them.
FLAG_EUR_ANNUAL_RATE = "eur-usd-annual-rate"
FLAG_DAC_WEATHER_FIXED = "dac-weather-fixed"
FLAG_DAC_PRODUCTIVITY_FIXED = "dac-productivity-fixed"


@dataclass(frozen=True)
class PaperValue:
    """One input as the paper or its data states it."""

    stage: str
    parameter: str
    value: float
    unit: str
    source: str
    status: str
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ChainInput:
    """One input in the units of the chain, with the paper values it comes from."""

    stage: str
    parameter: str
    value: float
    unit: str
    basis: str  # how the value follows from the paper values
    paper: tuple[PaperValue, ...]

    @property
    def source(self) -> str:
        return "; ".join(dict.fromkeys(p.source for p in self.paper))

    @property
    def status(self) -> str:
        return "; ".join(dict.fromkeys(p.status for p in self.paper))

    @property
    def flags(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(f for p in self.paper for f in p.flags))


_TABLE2 = f"{PAPER}, Suppl. Table 2"
_METHODS = f"{PAPER}, Methods, Process chain model"
_EUR = (FLAG_EUR_ANNUAL_RATE,)

PAPER_VALUES: tuple[PaperValue, ...] = (
    # PEM electrolysis
    PaperValue("electrolysis", "capex", 553.0, "EUR/kW_el", f"{_TABLE2}, ref. 41 (IEA 2019)", STATUS_TABLE, _EUR),
    PaperValue("electrolysis", "fom_share", 0.02, "1/a", f"{_TABLE2}, ref. 42 (IRENA 2018)", STATUS_TABLE),
    PaperValue("electrolysis", "efficiency", 0.70, "kWh_H2/kWh_el", f"{_TABLE2}, ref. 41 (IEA 2019)", STATUS_TABLE),
    PaperValue("electrolysis", "lifetime", 20, "a", f"{_TABLE2}, ref. 42 (IRENA 2018)", STATUS_TABLE),
    # CO2 hydrogenation (methanol synthesis)
    PaperValue(
        "methanol",
        "capex",
        210.0,
        "EUR/(t_MeOH/a)",
        f"{_TABLE2}; Suppl. Fig. 2, median of plants > 500 kt/a (review of 37 sources)",
        STATUS_TABLE,
        _EUR,
    ),
    PaperValue("methanol", "fom_share", 0.035, "1/a", _TABLE2, STATUS_TABLE),
    PaperValue("methanol", "lifetime", 20, "a", _TABLE2, STATUS_TABLE),
    PaperValue("methanol", "h2_demand", 3.2, "mol_H2/mol_MeOH", _METHODS, STATUS_TEXT),
    PaperValue("methanol", "co2_demand", 1.06, "mol_CO2/mol_MeOH", _METHODS, STATUS_TEXT),
    PaperValue("methanol", "elec_demand", 0.72, "kWh_el/kg_MeOH", _METHODS, STATUS_TEXT),
    # Direct air capture
    PaperValue("dac", "capex", 316.0, "EUR/(tCO2/a)", f"{_TABLE2}, ref. 2 (Wenzel et al. 2025)", STATUS_TABLE, _EUR),
    PaperValue("dac", "fom_share", 0.04, "1/a", f"{_TABLE2}, ref. 2 (Wenzel et al. 2025)", STATUS_TABLE),
    PaperValue("dac", "lifetime", 20, "a", f"{_TABLE2}, ref. 2 (Wenzel et al. 2025)", STATUS_TABLE),
    PaperValue(
        "dac",
        "elec_demand",
        0.4478,
        "MWh_el/tCO2",
        f"{FIGSHARE}, conversionFactor.DAC.electricity (scaled to 64.6 % of current demand)",
        STATUS_FIGSHARE,
        (FLAG_DAC_WEATHER_FIXED,),
    ),
    PaperValue(
        "dac",
        "heat_demand",
        2.0908,
        "MWh_th/tCO2",
        f"{FIGSHARE}, conversionFactor.DAC.heat (scaled to 64.7 % of current demand)",
        STATUS_FIGSHARE,
        (FLAG_DAC_WEATHER_FIXED,),
    ),
    PaperValue(
        "dac",
        "productivity",
        1.0,
        "-",
        f"set to 1; {FIGSHARE}, operationRateMax.DAC, median 1.005",
        STATUS_CHOICE,
        (FLAG_DAC_PRODUCTIVITY_FIXED,),
    ),
    # Air-source heat pump for the DAC heat
    PaperValue(
        "heat_pump", "capex", 665.0, "EUR/kW_th", f"{_TABLE2}, ref. 44 (Grosse et al. 2017)", STATUS_TABLE, _EUR
    ),
    PaperValue("heat_pump", "fom_share", 0.02, "1/a", f"{_TABLE2}, ref. 44 (Grosse et al. 2017)", STATUS_TABLE),
    PaperValue("heat_pump", "lifetime", 25, "a", f"{_TABLE2}, ref. 44 (Grosse et al. 2017)", STATUS_TABLE),
    PaperValue(
        "heat_pump",
        "elec_per_heat",
        0.43515,
        "kWh_el/kWh_th",
        f"{FIGSHARE}, conversionFactor.heatpump_air.electricity "
        f"(= 1/COP, COP = T_H/(T_H - T_C) * 0.5 with T_H = 100 C, T_C = ambient; {PAPER}, Methods, Eq. 6)",
        STATUS_FIGSHARE,
        (FLAG_DAC_WEATHER_FIXED,),
    ),
)


def paper_value(stage: str, parameter: str) -> PaperValue:
    for value in PAPER_VALUES:
        if value.stage == stage and value.parameter == parameter:
            return value
    raise KeyError(f"no paper value for {stage}.{parameter}")


def _usd(eur: PaperValue) -> float:
    return eur.value * USD_PER_EUR


def _lifetime(stage: str) -> ChainInput:
    p = paper_value(stage, "lifetime")
    return ChainInput(stage, "lifetime", p.value, "a", "as stated", (p,))


def _capex_and_fom(stage: str, capex_usd: float, unit: str, basis: str) -> tuple[ChainInput, ChainInput]:
    capex, share = paper_value(stage, "capex"), paper_value(stage, "fom_share")
    return (
        ChainInput(stage, "capex", capex_usd, unit, basis, (capex,)),
        ChainInput(
            stage,
            "fom",
            share.value * capex_usd,
            f"{unit}/a",
            f"fom_share {share.value} * capex",
            (share, capex),
        ),
    )


def chain_inputs() -> dict[tuple[str, str], ChainInput]:
    """Every input of the chain, keyed by ``(stage, parameter)``, in the units of the chain."""
    rate = f"* {USD_PER_EUR} USD/EUR"
    inputs: list[ChainInput] = []

    # Electrolysis: per kW_el.
    efficiency = paper_value("electrolysis", "efficiency")
    inputs += [
        *_capex_and_fom("electrolysis", _usd(paper_value("electrolysis", "capex")), "USD/kW_el", rate),
        ChainInput("electrolysis", "efficiency", efficiency.value, "kWh_H2/kWh_el", "as stated", (efficiency,)),
        _lifetime("electrolysis"),
    ]

    # Methanol synthesis: per kW_MeOH (LHV). One kW_MeOH of nameplate capacity makes 8760 kWh/a.
    tonnes_per_year_per_kw = HOURS_IN_YEAR / (KWH_PER_KG_METHANOL_LHV * KWH_PER_MWH)
    capex_per_kw = _usd(paper_value("methanol", "capex")) * tonnes_per_year_per_kw
    h2, co2, elec = (paper_value("methanol", name) for name in ("h2_demand", "co2_demand", "elec_demand"))
    h2_kwh_per_kg_methanol = h2.value * MOLAR_MASS_H2 / MOLAR_MASS_METHANOL * KWH_PER_KG_H2_LHV
    inputs += [
        *_capex_and_fom(
            "methanol", capex_per_kw, "USD/kW_MeOH", f"{rate} * 8760 h/a / {KWH_PER_KG_METHANOL_LHV:.4f} kWh/kg"
        ),
        ChainInput(
            "methanol",
            "efficiency",
            KWH_PER_KG_METHANOL_LHV / h2_kwh_per_kg_methanol,
            "kWh_MeOH/kWh_H2",
            "methanol LHV / (H2 mol/mol * M_H2/M_MeOH * H2 LHV)",
            (h2,),
        ),
        ChainInput(
            "methanol",
            "elec_demand",
            elec.value / KWH_PER_KG_METHANOL_LHV,
            "kWh_el/kWh_MeOH",
            "kWh/kg_MeOH / methanol LHV",
            (elec,),
        ),
        ChainInput(
            "methanol",
            "co2_demand",
            co2.value * MOLAR_MASS_CO2 / MOLAR_MASS_METHANOL / KWH_PER_KG_METHANOL_LHV,
            "kgCO2/kWh_MeOH",
            "CO2 mol/mol * M_CO2/M_MeOH / methanol LHV",
            (co2,),
        ),
        _lifetime("methanol"),
    ]

    # DAC: per (tCO2/a) of nameplate capacity; energy per captured tonne.
    dac_elec, dac_heat, productivity = (
        paper_value("dac", name) for name in ("elec_demand", "heat_demand", "productivity")
    )
    inputs += [
        *_capex_and_fom("dac", _usd(paper_value("dac", "capex")), "USD/(tCO2/a)", rate),
        ChainInput("dac", "elec_demand", dac_elec.value * KWH_PER_MWH, "kWh_el/tCO2", "MWh -> kWh", (dac_elec,)),
        ChainInput("dac", "heat_demand", dac_heat.value * KWH_PER_MWH, "kWh_th/tCO2", "MWh -> kWh", (dac_heat,)),
        ChainInput("dac", "productivity", productivity.value, "-", "as stated", (productivity,)),
        _lifetime("dac"),
    ]

    # Heat pump: per kW_th.
    elec_per_heat = paper_value("heat_pump", "elec_per_heat")
    inputs += [
        *_capex_and_fom("heat_pump", _usd(paper_value("heat_pump", "capex")), "USD/kW_th", rate),
        ChainInput(
            "heat_pump", "cop", 1.0 / elec_per_heat.value, "kWh_th/kWh_el", "1 / elec_per_heat", (elec_per_heat,)
        ),
        _lifetime("heat_pump"),
    ]

    return {(i.stage, i.parameter): i for i in inputs}
