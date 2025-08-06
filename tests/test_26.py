import pytest
from unittest.mock import patch
import matplotlib.pyplot as plt
from definition_fb14ce9e2bb142f282642c2653afe7bb import plot_mean_lgd_by_grade


@patch("matplotlib.pyplot.show")
def test_plot_mean_lgd_by_grade_no_errors(mock_show):
    """
    Test that the function executes without errors when mocked data is available.
    """
    try:
        plot_mean_lgd_by_grade()
    except Exception as e:
        pytest.fail(f"plot_mean_lgd_by_grade raised an exception: {e}")


@patch("matplotlib.pyplot.bar")
@patch("matplotlib.pyplot.show")
def test_plot_mean_lgd_by_grade_called_bar(mock_show, mock_bar):
    """
    Test that matplotlib.pyplot.bar is called within the function.
    """
    plot_mean_lgd_by_grade()
    assert mock_bar.called


@patch("matplotlib.pyplot.title")
@patch("matplotlib.pyplot.show")
def test_plot_mean_lgd_by_grade_called_title(mock_show, mock_title):
    """
    Test that matplotlib.pyplot.title is called within the function.
    """
    plot_mean_lgd_by_grade()
    assert mock_title.called

@patch("matplotlib.pyplot.xlabel")
@patch("matplotlib.pyplot.show")
def test_plot_mean_lgd_by_grade_called_xlabel(mock_show, mock_xlabel):
    """
    Test that matplotlib.pyplot.xlabel is called within the function.
    """
    plot_mean_lgd_by_grade()
    assert mock_xlabel.called


@patch("matplotlib.pyplot.ylabel")
@patch("matplotlib.pyplot.show")
def test_plot_mean_lgd_by_grade_called_ylabel(mock_show, mock_ylabel):
    """
    Test that matplotlib.pyplot.ylabel is called within the function.
    """
    plot_mean_lgd_by_grade()
    assert mock_ylabel.called
