import pytest
import pandas as pd
import numpy as np
from definition_2b5eca8cce6c4fcd966112a844ddedc7 import calibration_bins

def test_calibration_bins_empty():
    y_true = pd.Series([])
    y_pred = pd.Series([])
    n_bins = 10
    result = calibration_bins(y_true, y_pred, n_bins)
    assert result.empty

def test_calibration_bins_perfect_calibration():
    y_true = pd.Series([0, 0, 1, 1, 0, 1])
    y_pred = pd.Series([0.1, 0.2, 0.8, 0.9, 0.2, 0.7])
    n_bins = 3
    result = calibration_bins(y_true, y_pred, n_bins)
    assert not result.empty
    assert len(result) == n_bins

def test_calibration_bins_miscalibration():
    y_true = pd.Series([0, 0, 1, 1, 0, 1])
    y_pred = pd.Series([0.8, 0.9, 0.1, 0.2, 0.7, 0.2])
    n_bins = 3
    result = calibration_bins(y_true, y_pred, n_bins)
    assert not result.empty
    assert len(result) == n_bins

def test_calibration_bins_n_bins_greater_than_data():
    y_true = pd.Series([0, 1])
    y_pred = pd.Series([0.2, 0.8])
    n_bins = 5
    result = calibration_bins(y_true, y_pred, n_bins)
    assert not result.empty
    assert len(result) == n_bins

def test_calibration_bins_different_lengths():
    y_true = pd.Series([0, 1])
    y_pred = pd.Series([0.2, 0.8, 0.5])
    n_bins = 2
    with pytest.raises(ValueError):
        calibration_bins(y_true, y_pred, n_bins)
