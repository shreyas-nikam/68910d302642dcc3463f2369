import pytest
from definition_144eeac3992d4072a2a947af508c7257 import calculate_realized_lgd

@pytest.mark.parametrize("ead, recoveries, collection_costs, interest_rate, recovery_times, expected", [
    (1000, [200], [50], 0.05, [1], 0.761904761904762),
    (1000, [], [], 0.05, [], 1.0),
    (1000, [500, 300], [100, 50], 0.10, [0.5, 1], 0.2861332255205239),
    (1000, [1200], [0], 0.05, [1], 0),
    (1000, [200], [50], 0, [1], 0.75)
])
def test_calculate_realized_lgd(ead, recoveries, collection_costs, interest_rate, recovery_times, expected):
    assert calculate_realized_lgd(ead, recoveries, collection_costs, interest_rate, recovery_times) == pytest.approx(expected)
