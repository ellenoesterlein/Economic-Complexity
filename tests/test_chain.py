from dataclasses import replace

import numpy as np
import pytest

from lcom.chain import (
    DacParams,
    ElectrolysisParams,
    HeatPumpParams,
    MethanolParams,
    co2_cost,
    lcoh,
    lcom,
)
from lcom.constants import KWH_PER_KG_H2_LHV, KWH_PER_KG_METHANOL_LHV
from lcom.finance import crf
from lcom.paper_inputs import USD_PER_EUR, route_b_params

ELECTROLYSER = ElectrolysisParams(capex=1000.0, fom=20.0, efficiency=0.7, lifetime=25)
HEAT_PUMP = HeatPumpParams(capex=1000.0, fom=20.0, cop=2.0, lifetime=25)
DAC = DacParams(capex=1000.0, fom=100.0, elec_demand=500.0, heat_demand=2000.0, lifetime=15)
METHANOL = MethanolParams(capex=1500.0, fom=25.0, efficiency=0.8, elec_demand=0.1, co2_demand=0.25, lifetime=25)

# Results of the BOA branch feature/downstream-supply-chain (src/boa/downstream/chain.py) for the
# parameters above and these inputs. The port must reproduce them.
PARITY_E = np.array([30.0, 50.0, 90.0])
PARITY_W = np.array([0.05, 0.07, 0.10])
PARITY_CF = np.array([0.85, 0.85, 0.5])
PARITY = {
    "lcoh": {
        "total": [60.30705984022671, 91.72911960797083, 171.02676848989591],
        "electricity": [42.85714285714286, 71.42857142857143, 128.57142857142858],
        "capital": [13.612765684208126, 16.463396880523693, 35.93218271037861],
        "fom": [3.837151298875715, 3.837151298875715, 6.523157208088715],
    },
    "co2": {
        "total": [300.42081037542835, 350.23797298175504, 657.3850296605985],
        "electricity": [45.0, 75.0, 135.0],
        "capital": [132.40173973347294, 152.21890233979963, 313.2526095692744],
        "fom": [123.01907064195542, 123.01907064195542, 209.1324200913242],
    },
    "lcom": {
        "total": [171.13993874907527, 227.86496686646845, 430.5662724304947],
        "hydrogen": [75.38382480028338, 114.66139950996353, 213.78346061236988],
        "electricity": [3.0, 5.0, 9.0],
        "co2": [75.10520259385709, 87.55949324543876, 164.34625741514964],
        "capital": [14.293403968418533, 17.286566724549882, 37.72879184589754],
        "fom": [3.3575073865162506, 3.3575073865162506, 5.707762557077626],
    },
}


def _annuity(rate: float, years: int) -> float:
    return 1.0 / sum(1.0 / (1.0 + rate) ** t for t in range(1, years + 1))


def test_parity_with_the_boa_branch():
    hydrogen = lcoh(PARITY_E, PARITY_W, PARITY_CF, ELECTROLYSER)
    co2 = co2_cost(PARITY_E, PARITY_W, PARITY_CF, DAC, HEAT_PUMP)
    methanol = lcom(PARITY_E, PARITY_W, PARITY_CF, hydrogen, co2, METHANOL)

    for name, result in (("lcoh", hydrogen), ("co2", co2), ("lcom", methanol)):
        for field, expected in PARITY[name].items():
            np.testing.assert_allclose(getattr(result, field), expected, rtol=1e-12, err_msg=f"{name}.{field}")


def test_lcoh_by_hand():
    e, w, cf = 50.0, 0.07, 0.85
    annual = 1000.0 * _annuity(w, 25) + 20.0
    expected_capital_and_fom = annual / (0.7 * 8760 * cf) * 1000.0

    result = lcoh(e, w, cf, ELECTROLYSER)

    assert result.electricity == pytest.approx(50.0 / 0.7)
    assert result.capital + result.fom == pytest.approx(expected_capital_and_fom)
    assert result.total == pytest.approx(50.0 / 0.7 + expected_capital_and_fom)


def test_lcoh_without_plant_cost_is_electricity_over_efficiency():
    free = ElectrolysisParams(capex=0.0, fom=0.0, efficiency=0.7, lifetime=25)

    assert lcoh(50.0, 0.07, 0.9, free).total == pytest.approx(50.0 / 0.7)


def test_lcoh_in_usd_per_kg_uses_the_lower_heating_value():
    result = lcoh(50.0, 0.07, 0.85, ELECTROLYSER)

    assert KWH_PER_KG_H2_LHV == pytest.approx(33.33, abs=0.01)
    assert result.usd_per_kg() == pytest.approx(result.total / 1000.0 * 33.333, rel=1e-4)


def test_lcoh_rises_with_wacc_and_falls_with_load_factor():
    low, high = lcoh(50.0, np.array([0.05, 0.10]), 0.85, ELECTROLYSER).total
    assert high > low

    full = lcoh(50.0, 0.07, 1.0, ELECTROLYSER)
    part = lcoh(50.0, 0.07, 0.5, ELECTROLYSER)
    assert part.capital == pytest.approx(2.0 * full.capital)
    assert part.electricity == pytest.approx(full.electricity)


def test_co2_cost_by_hand():
    e, w, cf = 50.0, 0.07, 0.85
    electricity = e / 1000.0 * (500.0 + 2000.0 / 2.0)
    heat_pump_kw = 2000.0 / (8760 * cf)
    capital = 1000.0 * _annuity(w, 15) / cf + 1000.0 * _annuity(w, 25) * heat_pump_kw
    fom = 100.0 / cf + 20.0 * heat_pump_kw

    result = co2_cost(e, w, cf, DAC, HEAT_PUMP)

    assert result.electricity == pytest.approx(electricity)
    assert result.capital == pytest.approx(capital)
    assert result.fom == pytest.approx(fom)
    assert result.total == pytest.approx(electricity + capital + fom)


def test_productivity_scales_only_the_dac_plant_terms():
    e, w, cf, p = 50.0, 0.07, 0.85, 0.65
    base = co2_cost(e, w, cf, DAC, HEAT_PUMP)
    low = co2_cost(e, w, cf, replace(DAC, productivity=p), HEAT_PUMP)
    heat_pump_kw = 2000.0 / (8760 * cf)
    hp_capital = 1000.0 * _annuity(w, 25) * heat_pump_kw
    hp_fom = 20.0 * heat_pump_kw

    assert low.electricity == pytest.approx(base.electricity)
    assert low.capital - hp_capital == pytest.approx((base.capital - hp_capital) / p)
    assert low.fom - hp_fom == pytest.approx((base.fom - hp_fom) / p)


@pytest.mark.parametrize("bad", [0.0, -0.5, np.nan])
def test_nonpositive_productivity_is_rejected(bad):
    with pytest.raises(ValueError, match="productivity"):
        co2_cost(50.0, 0.07, 0.85, replace(DAC, productivity=bad), HEAT_PUMP)


def test_lcom_by_hand():
    e, w, cf = 50.0, 0.07, 0.85
    hydrogen = lcoh(e, w, cf, ELECTROLYSER)
    co2 = co2_cost(e, w, cf, DAC, HEAT_PUMP)
    plant = (1500.0 * _annuity(w, 25) + 25.0) / (8760 * cf) * 1000.0
    expected = hydrogen.total / 0.8 + 50.0 * 0.1 + co2.total * 0.25 + plant

    result = lcom(e, w, cf, hydrogen, co2, METHANOL)

    assert result.total == pytest.approx(expected)
    assert result.hydrogen == pytest.approx(hydrogen.total / 0.8)
    assert result.electricity == pytest.approx(5.0)
    assert result.co2 == pytest.approx(co2.total * 0.25)
    assert result.capital + result.fom == pytest.approx(plant)


def test_lcom_components_add_up_and_convert_to_tonnes():
    hydrogen = lcoh(50.0, 0.07, 0.85, ELECTROLYSER)
    co2 = co2_cost(50.0, 0.07, 0.85, DAC, HEAT_PUMP)

    result = lcom(50.0, 0.07, 0.85, hydrogen, co2, METHANOL)

    assert result.total == pytest.approx(
        result.hydrogen + result.electricity + result.co2 + result.capital + result.fom
    )
    assert KWH_PER_KG_METHANOL_LHV == pytest.approx(5.5278, abs=1e-4)
    assert result.usd_per_tonne() == pytest.approx(result.total * 5.5278, rel=1e-4)


def test_one_call_prices_a_grid():
    e = np.array([[30.0, 60.0], [45.0, 90.0]])
    w = np.array([[0.05, 0.05], [0.10, 0.10]])
    cf = np.full((2, 2), 0.85)

    hydrogen = lcoh(e, w, cf, ELECTROLYSER)
    co2 = co2_cost(e, w, cf, DAC, HEAT_PUMP)
    methanol = lcom(e, w, cf, hydrogen, co2, METHANOL)

    assert methanol.total.shape == (2, 2)
    assert methanol.total[0, 0] == pytest.approx(
        lcom(
            30.0, 0.05, 0.85, lcoh(30.0, 0.05, 0.85, ELECTROLYSER), co2_cost(30.0, 0.05, 0.85, DAC, HEAT_PUMP), METHANOL
        ).total
    )
    assert np.all(np.diff(methanol.total, axis=1) > 0)  # dearer power, dearer methanol


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.01, np.nan])
def test_load_factor_outside_zero_to_one_is_rejected(bad):
    with pytest.raises(ValueError, match="load_factor"):
        lcoh(50.0, 0.07, bad, ELECTROLYSER)


def test_route_b_per_tonne_in_paper_units():
    """Route B priced per tonne of methanol directly from the paper's units, independent of the conversions."""
    e, w, cf, r = 50.0, 0.07, 0.85, USD_PER_EUR
    cop = 1 / 0.43515
    h2_mwh_per_t = 3.2 * 2.016 / 32.042 * KWH_PER_KG_H2_LHV  # MWh_H2 per t_MeOH
    co2_t_per_t = 1.06 * 44.009 / 32.042  # tCO2 per t_MeOH

    electrolysis = e * h2_mwh_per_t / 0.7 + 553 * r * (crf(w, 20) + 0.02) * h2_mwh_per_t * 1000 / (0.7 * 8760 * cf)
    co2_per_t = (
        316 * r * (crf(w, 20) + 0.04) / cf
        + e / 1000 * (447.8 + 2090.8 / cop)
        + 665 * r * (crf(w, 25) + 0.02) * 2090.8 / (8760 * cf)
    )
    synthesis = 210 * r * (crf(w, 20) + 0.035) / cf + e * 0.72
    expected = electrolysis + co2_t_per_t * co2_per_t + synthesis

    p = route_b_params()
    hydrogen = lcoh(e, w, cf, p.electrolysis)
    co2 = co2_cost(e, w, cf, p.dac, p.heat_pump)
    result = lcom(e, w, cf, hydrogen, co2, p.methanol)

    assert co2.total == pytest.approx(co2_per_t, rel=1e-12)
    assert result.usd_per_tonne() == pytest.approx(expected, rel=1e-12)
