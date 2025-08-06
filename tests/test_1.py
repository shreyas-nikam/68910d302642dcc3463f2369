import pytest
import pandas as pd
from definition_f563577fe56047ecb0d94a137689dbc1 import assert_bounds

@pytest.mark.parametrize("x, lo, hi, expected_exception", [
    (pd.Series([0.2, 0.5, 0.8]), 0.0, 1.0, None),  # Within bounds
    (pd.Series([0.0, 1.0]), 0.0, 1.0, None),  # Exactly on bounds
    (pd.Series([0.1, 0.5, 1.1]), 0.0, 1.0, ValueError),  # Above upper bound
    (pd.Series([-0.1, 0.5, 0.8]), 0.0, 1.0, ValueError),  # Below lower bound
    (pd.Series([0.2, 0.5, 0.8]), 0.3, 0.7, ValueError), # Outside specified bounds
])
def test_assert_bounds(x, lo, hi, expected_exception):
    if expected_exception is None:
        assert_bounds(x, lo, hi)
    else:
        with pytest.raises(expected_exception):
            assert_bounds(x, lo, hi)
