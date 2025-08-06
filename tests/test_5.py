import pytest
from definition_334ad0c1721945afac49ef37bc5c2d8d import pv_cashflows

@pytest.mark.parametrize("cashflows, rate, time, expected", [
    ([100], 0.05, [1], 95.238),
    ([100, 100], 0.05, [1, 2], 185.941),
    ([50, 75, 100], 0.1, [0.5, 1, 1.5], 200.232),
    ([100], 0, [1], 100),
    ([100], -0.05, [1], ValueError)
])
def test_pv_cashflows(cashflows, rate, time, expected):
    if expected == ValueError:
        with pytest.raises(ValueError):
            pv_cashflows(cashflows, rate, time)
    else:
        assert round(pv_cashflows(cashflows, rate, time),3) == round(expected, 3)
