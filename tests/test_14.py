import pytest
import pandas as pd
from definition_42243c4a9ed641cdb604f80a3245294f import apply_lgd_floor

def test_apply_lgd_floor_empty_series():
    """Test with an empty Series."""
    s = pd.Series([])
    result = apply_lgd_floor(s)
    assert isinstance(result, pd.Series)
    assert result.empty

def test_apply_lgd_floor_no_values_below_floor():
    """Test with a Series where all values are above the floor."""
    s = pd.Series([0.1, 0.2, 0.3])
    result = apply_lgd_floor(s)
    assert (result == s).all()

def test_apply_lgd_floor_some_values_below_floor():
    """Test with a Series where some values are below the floor."""
    s = pd.Series([0.01, 0.06, 0.03])
    expected = pd.Series([0.05, 0.06, 0.05])
    result = apply_lgd_floor(s)
    pd.testing.assert_series_equal(result, expected)

def test_apply_lgd_floor_all_values_below_floor():
    """Test with a Series where all values are below the floor."""
    s = pd.Series([0.01, 0.02, 0.03])
    expected = pd.Series([0.05, 0.05, 0.05])
    result = apply_lgd_floor(s)
    pd.testing.assert_series_equal(result, expected)

def test_apply_lgd_floor_mixed_values():
    """Test with a Series with a mix of values, including negative values."""
    s = pd.Series([-0.01, 0.1, 0.03, 0.2, -0.05])
    expected = pd.Series([0.05, 0.1, 0.05, 0.2, 0.05])
    result = apply_lgd_floor(s)
    pd.testing.assert_series_equal(result, expected)
