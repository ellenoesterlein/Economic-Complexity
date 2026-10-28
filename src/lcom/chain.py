"""
Cost of hydrogen (LCOH), CO2 from direct air capture, and methanol (LCOM), from the LCOE.

Pure array arithmetic with no I/O. ``lcoe`` (USD per MWh of electricity), ``wacc`` (fraction) and
``load_factor`` broadcast against each other, so one call prices a whole grid. Every cost is
per unit of product, on a lower-heating-value (LHV) basis for hydrogen and methanol.

    LCOH = e / eta_el + (capex_el * CRF(w, n_el) + fom_el) / (eta_el * 8760 * CF)
    CO2  = (capex_dac * CRF(w, n_dac) + fom_dac) / (CF * p)
           + e * (elec_dac + heat_dac / COP) / 1000
           + (capex_hp * CRF(w, n_hp) + fom_hp) * heat_dac / (8760 * CF)
    LCOM = LCOH / eta_syn + e * elec_syn + co2_syn * CO2
           + (capex_syn * CRF(w, n_syn) + fom_syn) / (8760 * CF)

with ``e`` the LCOE, ``w`` the regional WACC, ``CF`` the plant load factor and ``p`` the DAC
relative productivity. ``fom`` is an absolute annual cost per unit of capacity.
"""

from dataclasses import dataclass

import numpy as np

from lcom.constants import HOURS_IN_YEAR, KWH_PER_KG_H2_LHV, KWH_PER_KG_METHANOL_LHV, KWH_PER_MWH
from lcom.finance import crf

Array = np.ndarray | float


@dataclass(frozen=True)
class ElectrolysisParams:
    capex: float  # USD/kW_el
    fom: float  # USD/kW_el/a
    efficiency: float  # kWh_H2 per kWh_el
    lifetime: int  # years


@dataclass(frozen=True)
class HeatPumpParams:
    capex: float  # USD/kW_th
    fom: float  # USD/kW_th/a
    cop: float  # kWh_th per kWh_el
    lifetime: int


@dataclass(frozen=True)
class DacParams:
    capex: float  # USD per (tCO2/a) of nameplate capacity
    fom: float  # USD per (tCO2/a) of nameplate capacity, per year
    elec_demand: float  # kWh_el/tCO2
    heat_demand: float  # kWh_th/tCO2
    lifetime: int
    # Output at full load relative to nameplate (the paper's relative productivity, weather-dependent).
    productivity: float = 1.0


@dataclass(frozen=True)
class MethanolParams:
    capex: float  # USD/kW_MeOH
    fom: float  # USD/kW_MeOH/a
    efficiency: float  # kWh_MeOH per kWh_H2
    elec_demand: float  # kWh_el per kWh_MeOH
    co2_demand: float  # kgCO2 per kWh_MeOH
    lifetime: int


@dataclass(frozen=True)
class Lcoh:
    """USD per MWh of hydrogen (LHV), split by cost component."""

    total: Array
    electricity: Array
    capital: Array
    fom: Array

    def usd_per_kg(self) -> Array:
        return self.total / KWH_PER_MWH * KWH_PER_KG_H2_LHV


@dataclass(frozen=True)
class Co2Cost:
    """USD per tonne of CO2 captured from air, split by cost component."""

    total: Array
    electricity: Array  # DAC electricity plus the electricity of the heat pump
    capital: Array  # DAC annuity plus heat-pump annuity
    fom: Array  # DAC and heat-pump fixed O&M


@dataclass(frozen=True)
class Lcom:
    """USD per MWh of methanol (LHV), split by cost component."""

    total: Array
    hydrogen: Array
    electricity: Array
    co2: Array
    capital: Array
    fom: Array

    def usd_per_tonne(self) -> Array:
        return self.total * KWH_PER_KG_METHANOL_LHV


def _check_load_factor(load_factor: Array) -> None:
    lf = np.asarray(load_factor, dtype=float)
    if np.any(~np.isfinite(lf)) or np.any(lf <= 0) or np.any(lf > 1):
        raise ValueError("load_factor must be in (0, 1].")


def lcoh(lcoe: Array, wacc: Array, load_factor: Array, p: ElectrolysisParams) -> Lcoh:
    _check_load_factor(load_factor)
    hours = HOURS_IN_YEAR * load_factor  # full-load hours of the electrolyser
    hydrogen_per_kw = p.efficiency * hours  # kWh_H2 per kW_el per year
    electricity = lcoe / p.efficiency
    capital = p.capex * crf(wacc, p.lifetime) / hydrogen_per_kw * KWH_PER_MWH
    fom = p.fom / hydrogen_per_kw * KWH_PER_MWH
    return Lcoh(total=electricity + capital + fom, electricity=electricity, capital=capital, fom=fom)


def co2_cost(lcoe: Array, wacc: Array, load_factor: Array, dac: DacParams, heat_pump: HeatPumpParams) -> Co2Cost:
    _check_load_factor(load_factor)
    if not (np.isfinite(dac.productivity) and dac.productivity > 0):
        raise ValueError("DAC productivity must be positive.")
    electricity_kwh = dac.elec_demand + dac.heat_demand / heat_pump.cop
    electricity = lcoe / KWH_PER_MWH * electricity_kwh
    # tCO2 captured per year per tCO2/a of DAC nameplate capacity.
    captured = load_factor * dac.productivity
    # Heat-pump capacity (kW_th) per tCO2/a actually captured. Heat is per captured tonne, and the heat
    # pump runs HOURS_IN_YEAR * CF hours, so productivity does not enter its size.
    heat_pump_kw = dac.heat_demand / (HOURS_IN_YEAR * load_factor)
    capital = (dac.capex * crf(wacc, dac.lifetime)) / captured + (
        heat_pump.capex * crf(wacc, heat_pump.lifetime) * heat_pump_kw
    )
    fom = dac.fom / captured + heat_pump.fom * heat_pump_kw
    return Co2Cost(total=electricity + capital + fom, electricity=electricity, capital=capital, fom=fom)


def lcom(lcoe: Array, wacc: Array, load_factor: Array, hydrogen: Lcoh, co2: Co2Cost, p: MethanolParams) -> Lcom:
    _check_load_factor(load_factor)
    methanol_kwh_per_kw = HOURS_IN_YEAR * load_factor  # kWh_MeOH per kW_MeOH per year
    hydrogen_cost = hydrogen.total / p.efficiency
    electricity = lcoe * p.elec_demand
    # kgCO2/kWh equals tCO2/MWh, so multiplying by USD/tCO2 gives USD/MWh directly.
    co2_part = co2.total * p.co2_demand
    capital = p.capex * crf(wacc, p.lifetime) / methanol_kwh_per_kw * KWH_PER_MWH
    fom = p.fom / methanol_kwh_per_kw * KWH_PER_MWH
    return Lcom(
        total=hydrogen_cost + electricity + co2_part + capital + fom,
        hydrogen=hydrogen_cost,
        electricity=electricity,
        co2=co2_part,
        capital=capital,
        fom=fom,
    )
