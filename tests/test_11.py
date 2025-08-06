import pytest
from definition_738445ca88aa4c8b83e2d03696af49f0 import compute_realized_lgd

@pytest.mark.parametrize("ead, pv_rec, pv_cost, expected", [
    (100.0, 20.0, 10.0, 0.7),
    (100.0, 0.0, 0.0, 1.0),
    (100.0, 100.0, 0.0, 0.0),
    (100.0, 50.0, 50.0, 0.0),
    (100.0, 20.0, 85.0, -0.05),
])
def test_compute_realized_lgd(ead, pv_rec, pv_cost, expected):
    assert compute_realized_lgd(ead, pv_rec, pv_cost) == (ead - pv_rec - pv_cost) / ead
