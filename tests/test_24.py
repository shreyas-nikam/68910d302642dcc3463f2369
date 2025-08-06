import pytest
import numpy as np
from definition_5b398579faf146c28d3d4f6ee18dfb1a import mae

@pytest.mark.parametrize("y_true, y_pred, expected", [
    (np.array([1, 2, 3]), np.array([1, 2, 3]), 0),
    (np.array([1, 2, 3]), np.array([4, 5, 6]), 3),
    (np.array([1, 2, 3]), np.array([0, 0, 0]), 2),
    (np.array([1.0, 2.0, 3.0]), np.array([1.5, 2.5, 3.5]), 0.5),
    (np.array([1, 2, 3]), np.array([3, 2, 1]), 4/3),
])
def test_mae(y_true, y_pred, expected):
    assert mae(y_true, y_pred) == expected
