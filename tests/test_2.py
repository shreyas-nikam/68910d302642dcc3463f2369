import pytest
import pandas as pd
from definition_72b9c43ad88d40b992a6a833f5a6467f import filter_defaults

@pytest.fixture
def sample_dataframe():
    data = {'loan_status': ['Fully Paid', 'Charged Off', 'Current', 'Charged Off'],
            'loan_amnt': [10000, 15000, 20000, 25000]}
    return pd.DataFrame(data)

def test_filter_defaults_empty_dataframe():
    df = pd.DataFrame()
    result = filter_defaults(df)
    assert isinstance(result, pd.DataFrame)
    assert result.empty

def test_filter_defaults_no_defaults(sample_dataframe):
    df = sample_dataframe[sample_dataframe['loan_status'] != 'Charged Off']
    result = filter_defaults(df)
    assert isinstance(result, pd.DataFrame)
    assert result.empty

def test_filter_defaults_some_defaults(sample_dataframe):
    result = filter_defaults(sample_dataframe)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert all(result['loan_status'] == 'Charged Off')
    assert len(result) == 2

def test_filter_defaults_all_defaults(sample_dataframe):
    df = sample_dataframe[sample_dataframe['loan_status'] == 'Charged Off']
    result = filter_defaults(df)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert all(result['loan_status'] == 'Charged Off')
    assert len(result) == len(df)
    assert result.equals(df)

def test_filter_defaults_invalid_input():
    with pytest.raises(TypeError):
        filter_defaults("not a dataframe")
