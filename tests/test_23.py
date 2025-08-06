import pytest
import pandas as pd
import numpy as np
from definition_518471874fa24d43a34ae033a10c2d7c import apply_pit_overlay

@pytest.fixture
def sample_data():
    ttc_pred = pd.Series([0.1, 0.2, 0.3])
    macro_row = pd.Series({'unemployment': 0.05, 'gdp_growth': 0.02})
    coefs = {'intercept': 0.01, 'unemployment': 0.02, 'gdp_growth': -0.01}
    return ttc_pred, macro_row, coefs

def test_apply_pit_overlay_additive(sample_data):
    ttc_pred, macro_row, coefs = sample_data
    expected = ttc_pred + coefs['intercept'] + macro_row['unemployment'] * coefs['unemployment'] + macro_row['gdp_growth'] * coefs['gdp_growth']
    result = apply_pit_overlay(ttc_pred, macro_row, coefs, mode='additive')
    assert np.allclose(result.values, expected.values)

def test_apply_pit_overlay_multiplicative(sample_data):
    ttc_pred, macro_row, coefs = sample_data
    adjustment = coefs['intercept'] + macro_row['unemployment'] * coefs['unemployment'] + macro_row['gdp_growth'] * coefs['gdp_growth']
    expected = ttc_pred * (1 + adjustment)
    result = apply_pit_overlay(ttc_pred, macro_row, coefs, mode='multiplicative')
    assert np.allclose(result.values, expected.values)

def test_apply_pit_overlay_empty_ttc(sample_data):
    _, macro_row, coefs = sample_data
    ttc_pred = pd.Series([])
    expected = pd.Series([])
    result = apply_pit_overlay(ttc_pred, macro_row, coefs, mode='additive')
    assert result.empty == expected.empty

def test_apply_pit_overlay_zero_ttc(sample_data):
    _, macro_row, coefs = sample_data
    ttc_pred = pd.Series([0, 0, 0])
    expected = pd.Series([coefs['intercept'] + macro_row['unemployment'] * coefs['unemployment'] + macro_row['gdp_growth'] * coefs['gdp_growth']]*3)
    result = apply_pit_overlay(ttc_pred, macro_row, coefs, mode='additive')
    assert np.allclose(result.values, expected.values)

def test_apply_pit_overlay_invalid_mode(sample_data):
    ttc_pred, macro_row, coefs = sample_data
    with pytest.raises(ValueError):
        apply_pit_overlay(ttc_pred, macro_row, coefs, mode='invalid')
