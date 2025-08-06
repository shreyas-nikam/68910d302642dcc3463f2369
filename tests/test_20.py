import pytest
import pandas as pd
import numpy as np
from definition_6c33e6091e3a47798db544a78b9037a5 import mae

def test_mae_valid_input():
    y_true = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    y_pred = pd.Series([0.11, 0.22, 0.29, 0.41, 0.49])
    expected_mae = 0.012  # Calculated manually
    assert np.isclose(mae(y_true, y_pred), expected_mae)

def test_mae_empty_series():
    y_true = pd.Series([])
    y_pred = pd.Series([])
    assert mae(y_true, y_pred) == 0.0

def test_mae_different_lengths():
    y_true = pd.Series([0.1, 0.2, 0.3])
    y_pred = pd.Series([0.1, 0.2])
    with pytest.raises(ValueError):
        mae(y_true, y_pred)

def test_mae_identical_series():
    y_true = pd.Series([0.1, 0.2, 0.3])
    y_pred = pd.Series([0.1, 0.2, 0.3])
    assert mae(y_true, y_pred) == 0.0

def test_mae_negative_values():
   y_true = pd.Series([-0.1, 0.2, 0.3])
   y_pred = pd.Series([0.1, 0.2, -0.3])
   expected_mae = 0.2 #Calculated manually abs(-0.1-0.1) + abs(0.2-0.2) + abs(0.3-(-0.3)) / 3
   assert np.isclose(mae(y_true, y_pred), expected_mae)
