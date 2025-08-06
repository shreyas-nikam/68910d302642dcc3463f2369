import pytest
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch
from definition_00d23e2661d143d083d9669486be60ef import plot_quarterly_lgd_vs_unrate

@pytest.fixture
def mock_plt_show():
    with patch("matplotlib.pyplot.show") as mock_show:
        yield mock_show

def test_plot_quarterly_lgd_vs_unrate_valid_data(mock_plt_show):
    lgd_q = pd.Series([0.05, 0.06, 0.07], index=pd.to_datetime(['2023-01-01', '2023-04-01', '2023-07-01']))
    unrate_q = pd.Series([3.5, 3.6, 3.7], index=pd.to_datetime(['2023-01-01', '2023-04-01', '2023-07-01']))
    plot_quarterly_lgd_vs_unrate(lgd_q, unrate_q)
    mock_plt_show.assert_called_once()

def test_plot_quarterly_lgd_vs_unrate_empty_series(mock_plt_show):
    lgd_q = pd.Series([])
    unrate_q = pd.Series([])
    plot_quarterly_lgd_vs_unrate(lgd_q, unrate_q)
    mock_plt_show.assert_called_once()  # Should still run without error, plotting an empty chart.

def test_plot_quarterly_lgd_vs_unrate_different_indices(mock_plt_show):
    lgd_q = pd.Series([0.05, 0.06], index=pd.to_datetime(['2023-01-01', '2023-04-01']))
    unrate_q = pd.Series([3.5, 3.6, 3.7], index=pd.to_datetime(['2023-01-01', '2023-04-01', '2023-07-01']))
    plot_quarterly_lgd_vs_unrate(lgd_q, unrate_q)
    mock_plt_show.assert_called_once()

def test_plot_quarterly_lgd_vs_unrate_non_datetime_index(mock_plt_show):
    lgd_q = pd.Series([0.05, 0.06], index=['Q1', 'Q2'])
    unrate_q = pd.Series([3.5, 3.6], index=['Q1', 'Q2'])
    with pytest.raises(AttributeError):
        plot_quarterly_lgd_vs_unrate(lgd_q, unrate_q) # Expect error because matplotlib will not work
                                                      # with string indexes when plotting time series.

def test_plot_quarterly_lgd_vs_unrate_missing_values(mock_plt_show):
    lgd_q = pd.Series([0.05, None, 0.07], index=pd.to_datetime(['2023-01-01', '2023-04-01', '2023-07-01']))
    unrate_q = pd.Series([3.5, 3.6, None], index=pd.to_datetime(['2023-01-01', '2023-04-01', '2023-07-01']))
    plot_quarterly_lgd_vs_unrate(lgd_q, unrate_q)
    mock_plt_show.assert_called_once() # Should still plot with missing values omitted.
