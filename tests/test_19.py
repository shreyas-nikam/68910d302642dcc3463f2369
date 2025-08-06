import pytest
import pandas as pd
from definition_60b263e7ac49471cb18fca2aa8b90a86 import apply_lgd_floor

@pytest.mark.parametrize("lgd, floor, expected", [
    (pd.Series([0.1, 0.02, 0.5]), 0.05, pd.Series([0.1, 0.05, 0.5])),
    (pd.Series([0.1, 0.02, 0.5]), 0.15, pd.Series([0.15, 0.15, 0.5])),
    (pd.Series([-0.1, 0.0, 1.1]), 0.05, pd.Series([0.05, 0.05, 1.1])),
    (pd.Series([0.1, 0.2, 0.3]), 0.0, pd.Series([0.1, 0.2, 0.3])),
    (pd.Series([]), 0.05, pd.Series([])),
])
def test_apply_lgd_floor(lgd, floor, expected):
    pd.testing.assert_series_equal(apply_lgd_floor(lgd, floor), expected, check_dtype=False)
