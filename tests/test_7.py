import pytest
import matplotlib.pyplot as plt
from definition_50dafb64aefd40959ea3d4724b501bcb import generate_calibration_plot
import numpy as np

def test_generate_calibration_plot_empty_input():
    """Test with empty input arrays."""
    with pytest.raises(ValueError):
        generate_calibration_plot(np.array([]), np.array([]), 10)

def test_generate_calibration_plot_different_lengths():
    """Test with input arrays of different lengths."""
    with pytest.raises(ValueError):
        generate_calibration_plot(np.array([0.1, 0.2]), np.array([0.3]), 10)

def test_generate_calibration_plot_valid_input():
    """Test with valid input data and check if a figure is returned."""
    predicted_lgd = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    actual_lgd = np.array([0.15, 0.25, 0.35, 0.45, 0.55])
    n_bins = 5
    fig = generate_calibration_plot(predicted_lgd, actual_lgd, n_bins)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_generate_calibration_plot_invalid_n_bins():
    """Test with invalid n_bins (less than or equal to 0)."""
    predicted_lgd = np.array([0.1, 0.2, 0.3])
    actual_lgd = np.array([0.15, 0.25, 0.35])
    with pytest.raises(ValueError):
        generate_calibration_plot(predicted_lgd, actual_lgd, 0)

def test_generate_calibration_plot_edge_case_n_bins():
    """Test with n_bins equal to the number of data points."""
    predicted_lgd = np.array([0.1, 0.2, 0.3])
    actual_lgd = np.array([0.15, 0.25, 0.35])
    n_bins = 3
    fig = generate_calibration_plot(predicted_lgd, actual_lgd, n_bins)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
