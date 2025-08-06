import pytest
from definition_5359329be6fb4b5898e5834da3bee043 import add_default_quarter
import pandas as pd

def test_add_default_quarter_empty_dataframe():
    """Test that the function returns an empty series if the input dataframe is empty."""
    df = pd.DataFrame()
    result = add_default_quarter()
    assert isinstance(result, pd.Series)
    assert result.empty

def test_add_default_quarter_no_default_date():
    """Test when 'default_date' column does not exist."""
    df = pd.DataFrame({'loan_id': [1, 2, 3]})
    with pytest.raises(KeyError):
        add_default_quarter()

def test_add_default_quarter_default_date_present():
    """Test when 'default_date' column exists and contains valid dates."""
    df = pd.DataFrame({'default_date': ['2023-01-15', '2023-04-20', '2023-07-01']})
    expected_quarters = pd.Series(['2023Q1', '2023Q2', '2023Q3'])
    pd.testing.assert_series_equal(add_default_quarter(), expected_quarters, check_names=False)

def test_add_default_quarter_invalid_date_format():
    """Test when 'default_date' column contains invalid date formats."""
    df = pd.DataFrame({'default_date': ['2023-01-15', 'invalid_date', '2023-07-01']})
    with pytest.raises(ValueError):
        add_default_quarter()

def test_add_default_quarter_mixed_date_formats():
    """Test when 'default_date' column contains mixed valid and invalid date formats."""
    df = pd.DataFrame({'default_date': ['2023-01-15', None, '2023-07-01']})

    expected_quarters = pd.Series(['2023Q1', None, '2023Q3'])
    pd.testing.assert_series_equal(add_default_quarter(), expected_quarters, check_names=False)
