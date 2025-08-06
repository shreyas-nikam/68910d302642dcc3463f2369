import pytest
import pandas as pd
from definition_e677f736cfef4b1aaadbd452a3e9c906 import filter_defaults

@pytest.fixture
def sample_dataframe():
    data = {'loan_status': ['Fully Paid', 'Charged Off', 'Current', 'Charged Off'],
            'loan_amnt': [10000, 20000, 15000, 25000]}
    return pd.DataFrame(data)

def test_filter_defaults_empty(sample_dataframe):
    df = sample_dataframe[sample_dataframe['loan_status'] != 'Charged Off']
    result = filter_defaults(df)
    assert len(result) == 0
    assert isinstance(result, pd.DataFrame)

def test_filter_defaults_some(sample_dataframe):
    result = filter_defaults(sample_dataframe)
    assert len(result) == 2
    assert all(result['loan_status'] == 'Charged Off')

def test_filter_defaults_all(sample_dataframe):
    df = sample_dataframe[sample_dataframe['loan_status'] == 'Charged Off']
    result = filter_defaults(df)
    assert len(result) == 2
    assert all(result['loan_status'] == 'Charged Off')

def test_filter_defaults_no_dataframe():
    with pytest.raises(TypeError):
        filter_defaults("not a dataframe")

def test_filter_defaults_column_missing(sample_dataframe):
    df = sample_dataframe.drop('loan_status', axis=1)
    with pytest.raises(KeyError):
        filter_defaults(df)
