import pytest
import pandas as pd
from definition_c19b2e08e0e245e983e7e17e6fdbe946 import apply_regulatory_floor

@pytest.fixture
def sample_series():
    return pd.Series([0.01, 0.05, 0.1, 0.03, 0.07])

def test_apply_regulatory_floor_below_floor(sample_series):
    floor = 0.05
    result = apply_regulatory_floor(sample_series, floor)
    expected = pd.Series([0.05, 0.05, 0.1, 0.05, 0.07])
    pd.testing.assert_series_equal(result, expected)

def test_apply_regulatory_floor_above_floor(sample_series):
    floor = 0.02
    result = apply_regulatory_floor(sample_series, floor)
    expected = pd.Series([0.02, 0.05, 0.1, 0.03, 0.07])
    pd.testing.assert_series_equal(result, expected)

def test_apply_regulatory_floor_empty_series():
    floor = 0.05
    empty_series = pd.Series([])
    result = apply_regulatory_floor(empty_series, floor)
    pd.testing.assert_series_equal(result, pd.Series([]))

def test_apply_regulatory_floor_negative_values():
    series = pd.Series([-0.01, 0.05, -0.1, 0.03, 0.07])
    floor = 0.05
    result = apply_regulatory_floor(series, floor)
    expected = pd.Series([0.05, 0.05, 0.05, 0.05, 0.07])
    pd.testing.assert_series_equal(result, expected)

def test_apply_regulatory_floor_zero_floor(sample_series):
    floor = 0.0
    result = apply_regulatory_floor(sample_series, floor)
    expected = sample_series
    pd.testing.assert_series_equal(result, expected)
