import pytest
from definition_bf10054010fb4e82aa442de6748deae1 import filter_defaults
import pandas as pd
import numpy as np

@pytest.fixture
def sample_loan_data():
    data = {'loan_status': ['Fully Paid', 'Charged Off', 'Current', 'Charged Off', 'Fully Paid'],
            'loan_amnt': [10000, 15000, 20000, 12000, 8000]}
    return pd.DataFrame(data)

def test_filter_defaults_empty_dataframe():
    df = pd.DataFrame()
    result = filter_defaults(df)
    assert isinstance(result, pd.DataFrame)
    assert result.empty

def test_filter_defaults_no_defaults(sample_loan_data):
    df = sample_loan_data[sample_loan_data['loan_status'] != 'Charged Off'].copy()
    result = filter_defaults(df)
    assert isinstance(result, pd.DataFrame)
    assert result.empty

def test_filter_defaults_some_defaults(sample_loan_data):
    result = filter_defaults(sample_loan_data)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert all(result['loan_status'] == 'Charged Off')
    assert len(result) == 2

def test_filter_defaults_all_defaults(sample_loan_data):
    df = sample_loan_data[sample_loan_data['loan_status'] == 'Charged Off'].copy()
    expected_length = len(df)
    result = filter_defaults(df)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert all(result['loan_status'] == 'Charged Off')
    assert len(result) == expected_length