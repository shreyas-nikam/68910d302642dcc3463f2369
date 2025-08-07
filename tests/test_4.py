import pytest
import pandas as pd
from definition_f535bed1f16f42239267c018aac12f92 import compute_ead

@pytest.fixture
def sample_dataframe():
    data = {'loan_amnt': [1000, 2000, 3000],
            'funded_amnt': [1000, 2000, 3000]}
    return pd.DataFrame(data)

def test_compute_ead_empty_dataframe():
    df = pd.DataFrame()
    result = compute_ead(df)
    assert isinstance(result, pd.DataFrame)
    assert result.empty

def test_compute_ead_same_funded_and_loan_amount(sample_dataframe):
    result = compute_ead(sample_dataframe)
    assert isinstance(result, pd.DataFrame)
    assert result.equals(sample_dataframe)

def test_compute_ead_with_one_row():
    data = {'loan_amnt': [1000],
            'funded_amnt': [900]}
    df = pd.DataFrame(data)
    result = compute_ead(df)
    assert isinstance(result, pd.DataFrame)
    assert result.equals(df)

def test_compute_ead_with_different_funded_and_loan_amount():
    data = {'loan_amnt': [1000, 2000],
            'funded_amnt': [900, 1800]}
    df = pd.DataFrame(data)
    result = compute_ead(df)
    assert isinstance(result, pd.DataFrame)
    assert result.equals(df)

def test_compute_ead_non_dataframe_input():
    with pytest.raises(AttributeError):
        compute_ead("not a dataframe")
