import numpy as np
import pytest
from boa.model.cost_calculations import _lcoe_tech_weights

from lcom.finance import annual_cost_per_kw, crf


def _annuity(rate: float, years: int) -> float:
    """Independent closed form: the inverse of the sum of year-end discount factors."""
    return 1.0 / sum(1.0 / (1.0 + rate) ** t for t in range(1, years + 1))


def test_crf_is_the_inverse_of_the_discounted_year_sum():
    assert crf(0.07, 25) == pytest.approx(_annuity(0.07, 25))
    assert crf(0.12, 15) == pytest.approx(_annuity(0.12, 15))


def test_crf_matches_the_closed_form():
    w, n = 0.08, 20
    assert crf(w, n) == pytest.approx(w / (1 - (1 + w) ** -n))


def test_crf_at_zero_rate_is_one_over_years():
    assert crf(0.0, 20) == pytest.approx(0.05)


def test_crf_rejects_a_negative_rate_and_a_nonpositive_life():
    with pytest.raises(ValueError, match="negative"):
        crf(-0.01, 10)
    with pytest.raises(ValueError, match="positive"):
        crf(0.05, 0)


def test_crf_works_on_arrays_with_a_zero_entry():
    result = crf(np.array([0.0, 0.05, 0.1]), 10)

    assert result == pytest.approx([0.1, _annuity(0.05, 10), _annuity(0.1, 10)])


def test_annual_cost_matches_the_lcoe_cost_weights():
    """Capex annuity plus fixed O&M equals BOA's discounted lifetime cost spread over its discounted energy."""
    wacc, life, opex_fraction, capex = 0.07, 25, 0.02, 1000.0
    discount = np.array([1.0 / (1.0 + wacc) ** t for t in range(life + 1)])
    w0, _ = _lcoe_tech_weights(discount, opex_fraction, life, life)
    boa_cost_per_kwh_year = w0 * capex / discount[1 : life + 1].sum()

    assert annual_cost_per_kw(capex, opex_fraction * capex, wacc, life) == pytest.approx(boa_cost_per_kwh_year)
