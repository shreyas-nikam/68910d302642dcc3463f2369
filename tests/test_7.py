import pytest
import pandas as pd
from definition_1b62d08c1aff42a79bf719d998bbc790 import filter_defaults

@pytest.fixture
def sample_dataframe():
    data = {'loan_id': [1, 2, 3, 4, 5],
            'status': ['Current', 'Default', 'Charged Off', 'Fully Paid', 'Default']}
    return pd.DataFrame(data)

def test_filter_defaults_empty_statuses(sample_dataframe):
    df = sample_dataframe
    default_statuses = set()
    result_df = filter_defaults(df, default_statuses)
    assert len(result_df) == 0
    
def test_filter_defaults_no_defaults(sample_dataframe):
    df = sample_dataframe
    default_statuses = {'NonExistentStatus'}
    result_df = filter_defaults(df, default_statuses)
    assert len(result_df) == 0

def test_filter_defaults_single_status(sample_dataframe):
    df = sample_dataframe
    default_statuses = {'Default'}
    result_df = filter_defaults(df, default_statuses)
    assert len(result_df) == 2
    assert all(result_df['status'] == 'Default')

def test_filter_defaults_multiple_statuses(sample_dataframe):
    df = sample_dataframe
    default_statuses = {'Default', 'Charged Off'}
    result_df = filter_defaults(df, default_statuses)
    assert len(result_df) == 3
    assert all(status in default_statuses for status in result_df['status'])

def test_filter_defaults_all_statuses(sample_dataframe):
    df = sample_dataframe
    default_statuses = {'Current', 'Default', 'Charged Off', 'Fully Paid'}
    result_df = filter_defaults(df, default_statuses)
    assert len(result_df) == 5
    pd.testing.assert_frame_equal(result_df, df)
