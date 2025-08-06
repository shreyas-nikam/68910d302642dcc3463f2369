import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from definition_52b85590ee2044d783616cae27554bfc import fit_pit_overlay
from statsmodels.regression.linear_model import RegressionResultsWrapper

def test_fit_pit_overlay_typical(monkeypatch):
    ttc_avg = pd.Series([0.1, 0.2, 0.3])
    macro_df = pd.DataFrame({'macro1': [1, 2, 3], 'macro2': [4, 5, 6]})

    mock_ols = MagicMock()
    mock_ols.fit.return_value = RegressionResultsWrapper(None) # Mock the results object

    monkeypatch.setattr('statsmodels.api.OLS', mock_ols)

    model = fit_pit_overlay(ttc_avg, macro_df)

    assert model is not None  # Check if the returned model is not None. Should be a fitted OLS model
    mock_ols.assert_called_once()  # Assert if OLS was called once.



def test_fit_pit_overlay_empty_ttc(monkeypatch):
    ttc_avg = pd.Series([])
    macro_df = pd.DataFrame({'macro1': [1, 2, 3], 'macro2': [4, 5, 6]})

    mock_ols = MagicMock()
    mock_ols.fit.return_value = RegressionResultsWrapper(None) # Mock the results object

    monkeypatch.setattr('statsmodels.api.OLS', mock_ols)

    model = fit_pit_overlay(ttc_avg, macro_df)
    assert model is not None

    mock_ols.assert_called_once()

def test_fit_pit_overlay_empty_macro(monkeypatch):
    ttc_avg = pd.Series([0.1, 0.2, 0.3])
    macro_df = pd.DataFrame()

    mock_ols = MagicMock()
    mock_ols.fit.return_value = RegressionResultsWrapper(None) # Mock the results object

    monkeypatch.setattr('statsmodels.api.OLS', mock_ols)

    model = fit_pit_overlay(ttc_avg, macro_df)
    assert model is not None
    mock_ols.assert_called_once()

def test_fit_pit_overlay_non_numeric_ttc(monkeypatch):
    ttc_avg = pd.Series(['a', 'b', 'c'])
    macro_df = pd.DataFrame({'macro1': [1, 2, 3], 'macro2': [4, 5, 6]})

    with pytest.raises(TypeError):
        fit_pit_overlay(ttc_avg, macro_df)

def test_fit_pit_overlay_non_numeric_macro(monkeypatch):
    ttc_avg = pd.Series([0.1, 0.2, 0.3])
    macro_df = pd.DataFrame({'macro1': ['a', 'b', 'c'], 'macro2': [4, 5, 6]})

    with pytest.raises(TypeError):
        fit_pit_overlay(ttc_avg, macro_df)
