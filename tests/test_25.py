import pytest
import pandas as pd
from definition_05f7fb93dc7341119387fa6756255421 import calibration_bins

@pytest.fixture
def sample_data():
    y_true = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    y_pred = pd.Series([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05])
    return y_true, y_pred

def test_calibration_bins_default(sample_data):
    y_true, y_pred = sample_data
    result = calibration_bins(y_true, y_pred, n_bins=10)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 10
    assert 'mean_predicted' in result.columns
    assert 'mean_actual' in result.columns

def test_calibration_bins_empty_input():
    y_true = pd.Series([])
    y_pred = pd.Series([])
    result = calibration_bins(y_true, y_pred, n_bins=5)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0

def test_calibration_bins_different_lengths():
    y_true = pd.Series([0.1, 0.2, 0.3])
    y_pred = pd.Series([0.15, 0.25, 0.35, 0.45])
    with pytest.raises(ValueError):
        calibration_bins(y_true, y_pred, n_bins=3)

def test_calibration_bins_non_series_input():
    with pytest.raises(TypeError):
        calibration_bins([0.1, 0.2], [0.15, 0.25], n_bins=2)

def test_calibration_bins_n_bins_greater_than_data_length(sample_data):
    y_true, y_pred = sample_data
    result = calibration_bins(y_true, y_pred, n_bins=12)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 12
    assert 'mean_predicted' in result.columns
    assert 'mean_actual' in result.columns