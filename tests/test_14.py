import pytest
from definition_b861e3ad78c14673842b27a537011e1e import apply_lgd_floor

@pytest.mark.parametrize("lgd_predictions, floor, expected", [
    ([0.1, 0.2, 0.3], 0.2, [0.2, 0.2, 0.3]),
    ([0.01, 0.02, 0.03], 0.05, [0.05, 0.05, 0.05]),
    ([0.5, 0.6, 0.7], 0.4, [0.5, 0.6, 0.7]),
    ([-0.1, 0.2, 0.3], 0.1, [0.1, 0.2, 0.3]),
    ([0.1, 0.2, 0.3], 0.0, [0.1, 0.2, 0.3]),
])
def test_apply_lgd_floor(lgd_predictions, floor, expected):
    assert apply_lgd_floor(lgd_predictions, floor) == expected
