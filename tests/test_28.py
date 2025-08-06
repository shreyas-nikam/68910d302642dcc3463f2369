import pytest
from unittest.mock import patch
import matplotlib.pyplot as plt
from definition_065424ec19824e19b5388d9c0c0ffdf5 import plot_calibration_curve


def test_plot_calibration_curve_success():
    """Tests that plot_calibration_curve runs without errors."""
    try:
        plot_calibration_curve()
    except Exception as e:
        pytest.fail(f"plot_calibration_curve raised an exception: {e}")


@patch("matplotlib.pyplot.show")
def test_plot_calibration_curve_calls_show(mock_show):
    """Tests that plot_calibration_curve calls plt.show()."""
    plot_calibration_curve()
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.figure")
def test_plot_calibration_curve_creates_figure(mock_figure):
    """Tests that plot_calibration_curve creates a matplotlib figure."""
    plot_calibration_curve()
    mock_figure.assert_called()

@patch("matplotlib.pyplot.savefig")
def test_plot_calibration_curve_saves_fig(mock_savefig):
    """Tests that plot_calibration_curve saves the plot."""
    plot_calibration_curve()
    assert mock_savefig.called