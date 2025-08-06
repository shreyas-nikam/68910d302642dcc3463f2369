import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from definition_ff7ecd8fc6684f19b114acf0c5df5a26 import train_pit_overlay_model
from sklearn.linear_model import LinearRegression

def test_train_pit_overlay_model_returns_trained_model():
    lgd_ttc = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    macroeconomic_factors = pd.DataFrame({'factor1': [1, 2, 3, 4, 5], 'factor2': [6, 7, 8, 9, 10]})
    lgd_realized = pd.Series([0.15, 0.25, 0.35, 0.45, 0.55])

    model = train_pit_overlay_model(lgd_ttc, macroeconomic_factors, lgd_realized)

    assert isinstance(model, LinearRegression)

def test_train_pit_overlay_model_handles_empty_input():
    lgd_ttc = pd.Series([])
    macroeconomic_factors = pd.DataFrame({'factor1': [], 'factor2': []})
    lgd_realized = pd.Series([])

    model = train_pit_overlay_model(lgd_ttc, macroeconomic_factors, lgd_realized)

    assert isinstance(model, LinearRegression)


def test_train_pit_overlay_model_different_index():
    lgd_ttc = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=[1,2,3,4,5])
    macroeconomic_factors = pd.DataFrame({'factor1': [1, 2, 3, 4, 5], 'factor2': [6, 7, 8, 9, 10]}, index=[1,2,3,4,5])
    lgd_realized = pd.Series([0.15, 0.25, 0.35, 0.45, 0.55], index=[1,2,3,4,5])

    model = train_pit_overlay_model(lgd_ttc, macroeconomic_factors, lgd_realized)

    assert isinstance(model, LinearRegression)

def test_train_pit_overlay_model_nan_values():

    lgd_ttc = pd.Series([0.1, 0.2, np.nan, 0.4, 0.5])
    macroeconomic_factors = pd.DataFrame({'factor1': [1, 2, 3, 4, 5], 'factor2': [6, np.nan, 8, 9, 10]})
    lgd_realized = pd.Series([0.15, 0.25, 0.35, 0.45, np.nan])

    model = train_pit_overlay_model(lgd_ttc, macroeconomic_factors, lgd_realized)

    assert isinstance(model, LinearRegression)

def test_train_pit_overlay_model_only_one_macro_factor():
    lgd_ttc = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    macroeconomic_factors = pd.DataFrame({'factor1': [1, 2, 3, 4, 5]})
    lgd_realized = pd.Series([0.15, 0.25, 0.35, 0.45, 0.55])

    model = train_pit_overlay_model(lgd_ttc, macroeconomic_factors, lgd_realized)

    assert isinstance(model, LinearRegression)
