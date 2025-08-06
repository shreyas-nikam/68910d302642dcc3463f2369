import pytest
from definition_3d3f18b956194a04b2ed46251f7d04d2 import plot_quarterly_lgd_vs_unrate
import matplotlib.pyplot as plt
from unittest.mock import patch

@patch('matplotlib.pyplot.show')
def test_plot_quarterly_lgd_vs_unrate_success(mock_show):
    try:
        plot_quarterly_lgd_vs_unrate()
    except Exception as e:
        assert False, f"plot_quarterly_lgd_vs_unrate raised an exception {e}"
    assert mock_show.called

@patch('matplotlib.pyplot.plot')
@patch('matplotlib.pyplot.twinx')
def test_plot_quarterly_lgd_vs_unrate_plot_called(mock_twinx, mock_plot):
    plot_quarterly_lgd_vs_unrate()
    assert mock_plot.called
    assert mock_twinx.called

@patch('matplotlib.pyplot.xlabel')
@patch('matplotlib.pyplot.ylabel')
def test_plot_quarterly_lgd_vs_unrate_labels_exist(mock_ylabel, mock_xlabel):
    plot_quarterly_lgd_vs_unrate()
    assert mock_xlabel.called
    assert mock_ylabel.called
    
@patch('matplotlib.pyplot.title')
def test_plot_quarterly_lgd_vs_unrate_title_exists(mock_title):
    plot_quarterly_lgd_vs_unrate()
    assert mock_title.called
