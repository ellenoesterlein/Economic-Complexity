import csv
import math
import statistics
from pathlib import Path

import pytest

from lcom import paper_inputs as pi

FIGSHARE_INPUTS = Path(__file__).parents[1] / "paper" / "figshare_30664253" / "mergedAveragedInputs_Year2018.csv"

EXPECTED_PARAMETERS = {
    "electrolysis": {"capex", "fom", "efficiency", "lifetime"},
    "methanol": {"capex", "fom", "efficiency", "elec_demand", "co2_demand", "lifetime"},
    "dac": {"capex", "fom", "elec_demand", "heat_demand", "productivity", "lifetime"},
    "heat_pump": {"capex", "fom", "cop", "lifetime"},
}


@pytest.fixture(scope="module")
def inputs():
    return pi.chain_inputs()


def test_every_paper_value_has_source_status_and_finite_value():
    keys = [(v.stage, v.parameter) for v in pi.PAPER_VALUES]
    assert len(keys) == len(set(keys))
    for v in pi.PAPER_VALUES:
        assert v.stage in pi.STAGES
        assert v.source.strip() and v.status.strip() and v.unit.strip()
        assert math.isfinite(v.value) and v.value > 0


def test_chain_inputs_are_complete(inputs):
    by_stage: dict[str, set[str]] = {}
    for stage, parameter in inputs:
        by_stage.setdefault(stage, set()).add(parameter)
    assert by_stage == EXPECTED_PARAMETERS
    for value in inputs.values():
        assert value.source and value.status and value.basis
        assert math.isfinite(value.value) and value.value > 0


def test_eur_values_convert_at_the_ecb_rate(inputs):
    assert pi.USD_PER_EUR == pytest.approx(1.08238046875)
    assert inputs["electrolysis", "capex"].value == pytest.approx(553 * pi.USD_PER_EUR)
    assert inputs["dac", "capex"].value == pytest.approx(316 * pi.USD_PER_EUR)
    assert inputs["heat_pump", "capex"].value == pytest.approx(665 * pi.USD_PER_EUR)
    assert pi.FLAG_EUR_ANNUAL_RATE in inputs["dac", "capex"].flags


@pytest.mark.parametrize(
    ("stage", "share", "capex_eur"),
    [("electrolysis", 0.02, 553), ("methanol", 0.035, None), ("dac", 0.04, 316), ("heat_pump", 0.02, 665)],
)
def test_fom_is_an_absolute_cost(inputs, stage, share, capex_eur):
    capex = inputs[stage, "capex"].value
    assert inputs[stage, "fom"].value == pytest.approx(share * capex)
    if capex_eur is not None:
        assert inputs[stage, "fom"].value == pytest.approx(share * capex_eur * pi.USD_PER_EUR)


def test_methanol_unit_conversions(inputs):
    # Hand-computed in the plan, section 4.
    assert inputs["methanol", "capex"].value / pi.USD_PER_EUR == pytest.approx(332.8, abs=0.05)
    assert inputs["methanol", "efficiency"].value == pytest.approx(0.8237, abs=5e-5)
    assert inputs["methanol", "elec_demand"].value == pytest.approx(0.13025, abs=5e-6)
    assert inputs["methanol", "co2_demand"].value == pytest.approx(0.26338, abs=5e-6)


def test_dac_and_heat_pump_values(inputs):
    assert inputs["dac", "elec_demand"].value == pytest.approx(447.8)
    assert inputs["dac", "heat_demand"].value == pytest.approx(2090.8)
    assert inputs["dac", "productivity"].value == 1.0
    assert inputs["heat_pump", "cop"].value == pytest.approx(2.298, abs=5e-4)
    assert pi.FLAG_DAC_WEATHER_FIXED in inputs["dac", "heat_demand"].flags


@pytest.mark.skipif(not FIGSHARE_INPUTS.exists(), reason="figshare data is local only (paper/)")
def test_figshare_medians_match_the_data():
    with FIGSHARE_INPUTS.open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20265

    def median(column: str) -> float:
        return statistics.median(float(r[column]) for r in rows)

    # The data stores consumption as negative conversion factors.
    assert -median("conversionFactor.DAC.electricity") == pytest.approx(
        pi.paper_value("dac", "elec_demand").value, abs=5e-5
    )
    assert -median("conversionFactor.DAC.heat") == pytest.approx(pi.paper_value("dac", "heat_demand").value, abs=5e-5)
    assert -median("conversionFactor.heatpump_air.electricity") == pytest.approx(
        pi.paper_value("heat_pump", "elec_per_heat").value, abs=5e-6
    )
    assert median("operationRateMax.DAC") == pytest.approx(1.005, abs=5e-4)
