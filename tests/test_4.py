import pytest
import pandas as pd
from unittest.mock import MagicMock
from definition_5216013068844d1f88725fd46abd2666 import predict_lgd


def test_predict_lgd_empty_data():
    model_mock = MagicMock()
    empty_data = pd.DataFrame()
    result = predict_lgd(model_mock, empty_data)
    assert isinstance(result, pd.Series)
    assert result.empty


def test_predict_lgd_model_predict_returns_series():
    model_mock = MagicMock()
    model_mock.predict.return_value = pd.Series([0.2, 0.5, 0.8])
    data = pd.DataFrame({'feature1': [1, 2, 3]})
    result = predict_lgd(model_mock, data)
    assert isinstance(result, pd.Series)
    assert len(result) == 3
    assert result.iloc[0] == 0.2
    assert result.iloc[1] == 0.5
    assert result.iloc[2] == 0.8


def test_predict_lgd_model_predict_raises_exception():
    model_mock = MagicMock()
    model_mock.predict.side_effect = ValueError("Prediction failed")
    data = pd.DataFrame({'feature1': [1, 2, 3]})
    with pytest.raises(ValueError, match="Prediction failed"):
        predict_lgd(model_mock, data)


def test_predict_lgd_model_predict_returns_list():
    model_mock = MagicMock()
    model_mock.predict.return_value = [0.2, 0.5, 0.8]
    data = pd.DataFrame({'feature1': [1, 2, 3]})
    result = predict_lgd(model_mock, data)
    assert isinstance(result, pd.Series)
    assert len(result) == 3
    assert result.iloc[0] == 0.2
    assert result.iloc[1] == 0.5
    assert result.iloc[2] == 0.8


def test_predict_lgd_with_sample_data():
    model_mock = MagicMock()
    model_mock.predict.return_value = pd.Series([0.1, 0.6, 0.9])
    data = pd.DataFrame({'feature1': [4, 5, 6]})
    expected_output = pd.Series([0.1, 0.6, 0.9])

    predicted_lgd = predict_lgd(model_mock, data)

    pd.testing.assert_series_equal(predicted_lgd, expected_output)

