import pytest
from definition_f617741388b34c27958974227c228ccc import apply_lgd_floor

@pytest.mark.parametrize("lgd_predictions, floor, expected", [
    ([0.1, 0.2, 0.03, 0.5], 0.05, [0.1, 0.2, 0.05, 0.5]),
    ([0.02, 0.04, 0.01], 0.05, [0.05, 0.05, 0.05]),
    ([0.6, 0.7, 0.8], 0.5, [0.6, 0.7, 0.8]),
    ([-0.1, 0.2, -0.03, 0.5], 0.05, [0.05, 0.2, 0.05, 0.5]),
    ([], 0.05, [])
])
def test_apply_lgd_floor(lgd_predictions, floor, expected):
    assert apply_lgd_floor(lgd_predictions, floor) == expected
