import pytest
from definition_34aa990434294eeca09dd3cb6c92f17c import compute_realized_lgd

@pytest.mark.parametrize("ead, pv_recoveries, pv_collection_costs, expected", [
    (1000, 200, 50, 0.75),
    (500, 0, 0, 1.0),
    (2000, 1000, 200, 0.4),
    (1000, 1000, 0, 0.0),
    (1000, 1200, 0, -0.2),  # Recovery exceeds EAD
])
def test_compute_realized_lgd(ead, pv_recoveries, pv_collection_costs, expected):
    assert compute_realized_lgd(ead, pv_recoveries, pv_collection_costs) == expected
