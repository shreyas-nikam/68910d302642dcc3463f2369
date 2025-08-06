import pytest
from definition_895061c218cb4c04aecd2553fd18adf6 import calculate_lgd

@pytest.mark.parametrize("ead, recoveries, expected", [
    (100, 20, 0.8),
    (100, 120, 0.0),
    (100, 0, 1.0),
    (100, 100, 0.0),
    (0, 0, 0.0),  # Edge case: EAD is zero
])
def test_calculate_lgd(ead, recoveries, expected):
    assert calculate_lgd(ead, recoveries) == expected
