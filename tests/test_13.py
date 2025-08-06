import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from definition_94d2680070b54e31a27c2da1afc4a236 import predict_beta


def test_predict_beta_typical(mocker):
    # Mock the model's predict method to return a known series of values
    mock_model = MagicMock()
    mock_model.predict.return_value = pd.Series([0.1, 0.2, 0.3])

    # Create a sample input dataframe
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})

    # Call the function with the mocked model and sample data
    result = predict_beta(mock_model, X)

    # Assert that the function returns a Pandas Series
    assert isinstance(result, pd.Series)

    # Assert that the predicted values are correct (as defined by the mock)
    assert np.allclose(result.values, [0.1, 0.2, 0.3])


def test_predict_beta_empty_dataframe(mocker):
    # Mock the model's predict method to return an empty series
    mock_model = MagicMock()
    mock_model.predict.return_value = pd.Series([])

    # Create an empty dataframe
    X = pd.DataFrame()

    # Call the function with the mocked model and empty data
    result = predict_beta(mock_model, X)

    # Assert that the function returns a Pandas Series
    assert isinstance(result, pd.Series)

    # Assert that the result is an empty series
    assert result.empty


def test_predict_beta_model_error(mocker):
    # Mock the model's predict method to raise an exception
    mock_model = MagicMock()
    mock_model.predict.side_effect = ValueError("Model failed to predict")

    # Create a sample input dataframe
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})

    # Call the function and expect the exception to propagate
    with pytest.raises(ValueError, match="Model failed to predict"):
        predict_beta(mock_model, X)


def test_predict_beta_all_zeros(mocker):
    # Mock the model's predict method to return all zeros
    mock_model = MagicMock()
    mock_model.predict.return_value = pd.Series([0, 0, 0])

    # Create a sample input dataframe
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})

    # Call the function with the mocked model and sample data
    result = predict_beta(mock_model, X)

    # Assert that the function returns a Pandas Series
    assert isinstance(result, pd.Series)

    # Assert that the predicted values are all zeros
    assert np.allclose(result.values, [0, 0, 0])



def test_predict_beta_nan_values(mocker):
    # Mock the model's predict method to return NaN values
    mock_model = MagicMock()
    mock_model.predict.return_value = pd.Series([np.nan, np.nan, np.nan])

    # Create a sample input dataframe
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})

    # Call the function with the mocked model and sample data
    result = predict_beta(mock_model, X)

    # Assert that the function returns a Pandas Series
    assert isinstance(result, pd.Series)

    # Assert that the predicted values are NaN
    assert np.isnan(result.values).all()
