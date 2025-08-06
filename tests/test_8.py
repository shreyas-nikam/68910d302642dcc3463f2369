import pytest
from definition_a7584b6f63ac469aa155a50dffd19239 import calculate_mean_absolute_error

@pytest.mark.parametrize("predicted_lgd, actual_lgd, expected", [
    ([0.1, 0.2, 0.3], [0.15, 0.25, 0.35], 0.05),
    ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0),
    ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], 0.0),
    ([0.2, 0.4, 0.6], [0.1, 0.3, 0.7], 0.06666666666666667),
    ([0.0, 1.0], [1.0, 0.0], 1.0)
])
def test_calculate_mean_absolute_error(predicted_lgd, actual_lgd, expected):
    assert calculate_mean_absolute_error(predicted_lgd, actual_lgd) == expected
