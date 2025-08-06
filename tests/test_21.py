import pytest
import pandas as pd
from definition_24928f375a86473392cb763d0ca4c4fd import align_macro_with_cohorts

@pytest.fixture
def mock_macro_df():
    return pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2023-04-01', '2023-07-01', '2023-10-01']),
        'unemployment_rate': [4.0, 4.1, 4.2, 4.3]
    })

@pytest.fixture
def mock_cohorts_df():
    return pd.DataFrame({
        'loan_id': [1, 2, 3, 4],
        'default_quarter': ['2023-01-01', '2023-04-01', '2023-07-01', '2023-10-01']
    })

def test_align_macro_with_cohorts_no_lag(mock_macro_df, mock_cohorts_df):
    # Test case 1: No lag
    result = align_macro_with_cohorts(mock_macro_df, mock_cohorts_df, lag_q=0)
    assert result is not None

def test_align_macro_with_cohorts_with_lag(mock_macro_df, mock_cohorts_df):
        # Test case 2: With lag
    result = align_macro_with_cohorts(mock_macro_df, mock_cohorts_df, lag_q=1)
    assert result is not None

def test_align_macro_with_cohorts_empty_macro(mock_cohorts_df):
    # Test case 3: Empty macro dataframe
    macro_df = pd.DataFrame({'date': [], 'unemployment_rate': []})
    result = align_macro_with_cohorts(macro_df, mock_cohorts_df, lag_q=1)
    assert result is not None

def test_align_macro_with_cohorts_empty_cohorts(mock_macro_df):
    # Test case 4: Empty cohorts dataframe
    cohorts_df = pd.DataFrame({'loan_id': [], 'default_quarter': []})
    result = align_macro_with_cohorts(mock_macro_df, cohorts_df, lag_q=1)
    assert result is not None

def test_align_macro_with_cohorts_invalid_lag(mock_macro_df, mock_cohorts_df):
    # Test case 5: Invalid lag value
    result = align_macro_with_cohorts(mock_macro_df, mock_cohorts_df, lag_q=-1)
    assert result is not None
