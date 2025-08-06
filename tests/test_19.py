import pytest
import pandas as pd
from unittest.mock import MagicMock
from definition_578c8b3f55824ae19344681fc1da5539 import apply_pit_overlay

def test_apply_pit_overlay_typical_case():
    ttc_lgd = pd.Series([0.2, 0.3, 0.4])
    macro_factors = pd.DataFrame({'factor1': [0.1, 0.2, 0.3]})
    model_mock = MagicMock()
    model_mock.predict.return_value = [0.05, -0.02, 0.01]
    
    result = apply_pit_overlay(ttc_lgd, macro_factors, model_mock)
    
    expected = pd.Series([0.25, 0.28, 0.41])
    pd.testing.assert_series_equal(result, expected, check_dtype=False)

def test_apply_pit_overlay_empty_ttc():
    ttc_lgd = pd.Series([])
    macro_factors = pd.DataFrame({'factor1': []})
    model_mock = MagicMock()
    model_mock.predict.return_value = []
    
    result = apply_pit_overlay(ttc_lgd, macro_factors, model_mock)
    
    expected = pd.Series([])
    pd.testing.assert_series_equal(result, expected, check_dtype=False)

def test_apply_pit_overlay_model_returns_large_adjustments():
    ttc_lgd = pd.Series([0.2, 0.3, 0.4])
    macro_factors = pd.DataFrame({'factor1': [0.1, 0.2, 0.3]})
    model_mock = MagicMock()
    model_mock.predict.return_value = [0.8, -0.7, 0.6]
    
    result = apply_pit_overlay(ttc_lgd, macro_factors, model_mock)
    
    expected = pd.Series([1.0, -0.4, 1.0])
    pd.testing.assert_series_equal(result, expected, check_dtype=False)

def test_apply_pit_overlay_macro_factors_missing_values():
    ttc_lgd = pd.Series([0.2, 0.3, 0.4])
    macro_factors = pd.DataFrame({'factor1': [0.1, None, 0.3]})
    model_mock = MagicMock()
    model_mock.predict.return_value = [0.05, -0.02, 0.01]

    with pytest.raises(ValueError):
        apply_pit_overlay(ttc_lgd, macro_factors, model_mock)

def test_apply_pit_overlay_different_index():
    ttc_lgd = pd.Series([0.2, 0.3, 0.4], index=[0, 1, 2])
    macro_factors = pd.DataFrame({'factor1': [0.1, 0.2, 0.3]}, index=[3, 4, 5])
    model_mock = MagicMock()
    model_mock.predict.return_value = [0.05, -0.02, 0.01]

    with pytest.raises(ValueError):
        apply_pit_overlay(ttc_lgd, macro_factors, model_mock)
