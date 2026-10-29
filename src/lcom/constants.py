"""Physical constants of the chain. Hydrogen and methanol use the lower heating value (LHV)."""

from boa.config.constants import HOURS_IN_YEAR

__all__ = [
    "HOURS_IN_YEAR",
    "KWH_PER_KG_H2_LHV",
    "KWH_PER_KG_METHANOL_LHV",
    "KWH_PER_MWH",
    "MOLAR_MASS_CO2",
    "MOLAR_MASS_H2",
    "MOLAR_MASS_METHANOL",
]

# Hydrogen 120 MJ/kg = 33.33 kWh/kg; methanol 19.9 MJ/kg = 5.528 kWh/kg.
KWH_PER_KG_H2_LHV = 120.0 / 3.6
KWH_PER_KG_METHANOL_LHV = 19.9 / 3.6
KWH_PER_MWH = 1000.0

# g/mol
MOLAR_MASS_H2 = 2.016
MOLAR_MASS_CO2 = 44.009
MOLAR_MASS_METHANOL = 32.042
