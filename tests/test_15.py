import pytest
import pandas as pd
from definition_c723cf39db534544883a77b81ed8135b import add_default_quarter

def test_add_default_quarter_typical(mocker):
    df = pd.DataFrame({'default_date': ['2023-01-15', '2023-04-20', '2023-07-01', '2023-10-31']})
    df['default_date'] = pd.to_datetime(df['default_date'])
    expected = pd.Series(['2023Q1', '2023Q2', '2023Q3', '2023Q4'])
    
    actual = add_default_quarter(df, 'default_date')
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_add_default_quarter_empty_df():
    df = pd.DataFrame({'default_date': []})
    expected = pd.Series([], dtype='object')
    actual = add_default_quarter(df, 'default_date')
    pd.testing.assert_series_equal(actual, expected, check_names=False)

def test_add_default_quarter_date_format_error(mocker):
    df = pd.DataFrame({'default_date': ['2023/01/15']})
    with pytest.raises(TypeError):
        add_default_quarter(df, 'default_date')

def test_add_default_quarter_missing_date_column():
    df = pd.DataFrame({'other_column': ['2023-01-15']})
    with pytest.raises(KeyError):
        add_default_quarter(df, 'default_date')

def test_add_default_quarter_mixed_dates(mocker):
    df = pd.DataFrame({'default_date': ['2023-01-15', '2023-04-20', None, '2023-10-31']})
    df['default_date'] = pd.to_datetime(df['default_date'], errors='coerce')
    expected = pd.Series(['2023Q1', '2023Q2', None, '2023Q4'])
    actual = add_default_quarter(df, 'default_date')
    pd.testing.assert_series_equal(actual, expected, check_names=False)
