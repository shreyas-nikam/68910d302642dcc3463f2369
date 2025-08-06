import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from definition_cc81eb9932394525914eef681ac601e2 import residuals_vs_fitted

def test_residuals_vs_fitted_valid_input():
    model_mock = MagicMock()
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y = pd.Series([7, 8, 9])
    
    model_mock.predict.return_value = np.array([7.1, 8.1, 9.1]) # Mock the fitted values
    residuals = y - model_mock.predict(X)
    expected_df = pd.DataFrame({'residuals': residuals, 'fitted': model_mock.predict(X)})

    result_df = residuals_vs_fitted(model_mock, X, y)
    
    pd.testing.assert_frame_equal(result_df, expected_df)

def test_residuals_vs_fitted_empty_dataframe():
    model_mock = MagicMock()
    X = pd.DataFrame()
    y = pd.Series([])

    result_df = residuals_vs_fitted(model_mock, X, y)

    assert isinstance(result_df, pd.DataFrame)
    assert result_df.empty

def test_residuals_vs_fitted_model_predict_error():
    model_mock = MagicMock()
    X = pd.DataFrame({'feature1': [1, 2, 3]})
    y = pd.Series([4, 5, 6])
    model_mock.predict.side_effect = ValueError("Prediction failed")

    with pytest.raises(ValueError, match="Prediction failed"):
        residuals_vs_fitted(model_mock, X, y)

def test_residuals_vs_fitted_y_mismatch_x():
    model_mock = MagicMock()
    X = pd.DataFrame({'feature1': [1, 2, 3]})
    y = pd.Series([4, 5])

    with pytest.raises(ValueError, match="arrays must all be same length"):
        residuals_vs_fitted(model_mock, X, y)

def test_residuals_vs_fitted_different_fitted_values():
    model_mock = MagicMock()
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    y = pd.Series([7, 8, 9])

    model_mock.predict.return_value = np.array([10, 11, 12])  # Different fitted values

    result_df = residuals_vs_fitted(model_mock, X, y)

    assert not result_df['fitted'].equals(y)

