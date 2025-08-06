import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from definition_746a14731f474e258b71e52d024e7642 import residuals_vs_fitted


def test_residuals_vs_fitted_valid_input():
    model_mock = MagicMock()
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y = pd.Series([7, 8, 9])
    model_mock.predict.return_value = np.array([7.1, 7.9, 9.2])

    result = residuals_vs_fitted(model_mock, X, y)
    
    assert isinstance(result, pd.DataFrame)
    assert 'residuals' in result.columns
    assert 'fitted_values' in result.columns
    assert len(result) == len(y)


def test_residuals_vs_fitted_empty_input():
    model_mock = MagicMock()
    X = pd.DataFrame()
    y = pd.Series()
    model_mock.predict.return_value = np.array([])

    result = residuals_vs_fitted(model_mock, X, y)

    assert isinstance(result, pd.DataFrame)
    assert 'residuals' in result.columns
    assert 'fitted_values' in result.columns
    assert len(result) == 0

def test_residuals_vs_fitted_unequal_lengths():
    model_mock = MagicMock()
    X = pd.DataFrame({'feature1': [1, 2, 3]})
    y = pd.Series([7, 8])
    model_mock.predict.return_value = np.array([7.1, 7.9])

    with pytest.raises(ValueError):
        residuals_vs_fitted(model_mock, X, y)

def test_residuals_vs_fitted_non_numeric_data():
    model_mock = MagicMock()
    X = pd.DataFrame({'feature1': ['a', 'b', 'c']})
    y = pd.Series([7, 8, 9])
    model_mock.predict.return_value = np.array([7.1, 7.9, 9.2])
    with pytest.raises(TypeError):
        residuals_vs_fitted(model_mock, X, y)

def test_residuals_vs_fitted_model_returns_none():
    model_mock = MagicMock()
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y = pd.Series([7, 8, 9])
    model_mock.predict.return_value = None

    with pytest.raises(TypeError):
        residuals_vs_fitted(model_mock, X, y)
