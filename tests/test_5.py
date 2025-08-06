import pytest
from definition_95f21b6f76f4451da29faed64e91dda0 import pv_cashflows

@pytest.mark.parametrize("cashflows, interest_rate, time_to_payment, expected", [
    ([100, 100, 100], 0.05, [1, 2, 3], 272.32),
    ([500], 0.1, [5], 310.46),
    ([200, 300], 0.0, [1, 2], 500.00),
    ([100, 50], 0.02, [0.5, 1], 147.56),
    ([100], -0.5, [1], ValueError)
])
def test_pv_cashflows(cashflows, interest_rate, time_to_payment, expected):
    if expected == ValueError:
        with pytest.raises(ValueError):
            pv_cashflows(cashflows, interest_rate, time_to_payment)
    else:
        result = pv_cashflows(cashflows, interest_rate, time_to_payment)
        assert round(result, 2) == round(expected, 2)
