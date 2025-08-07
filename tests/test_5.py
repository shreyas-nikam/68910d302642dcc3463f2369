import pytest
import pandas as pd
from definition_9a8efdb122c84d9b81a2da31cbd53986 import pv_cashflows

@pytest.fixture
def sample_dataframe():
    data = {'cashflow': [100, 110, 121], 'time': [1, 2, 3]}
    return pd.DataFrame(data)

def test_pv_cashflows_positive_rate(sample_dataframe):
    df = pv_cashflows(sample_dataframe.copy(), 0.1)
    assert 'present_value' in df.columns
    assert df['present_value'].sum() > 0

def test_pv_cashflows_zero_rate(sample_dataframe):
    df = pv_cashflows(sample_dataframe.copy(), 0.0)
    assert 'present_value' in df.columns
    assert df['present_value'].sum() == sample_dataframe['cashflow'].sum()

def test_pv_cashflows_negative_cashflows():
    data = {'cashflow': [-100, -110, -121], 'time': [1, 2, 3]}
    df = pd.DataFrame(data)
    df_pv = pv_cashflows(df.copy(), 0.1)
    assert 'present_value' in df_pv.columns
    assert df_pv['present_value'].sum() < 0

def test_pv_cashflows_empty_dataframe():
    df = pd.DataFrame({'cashflow': [], 'time': []})
    df_pv = pv_cashflows(df.copy(), 0.1)
    assert 'present_value' in df_pv.columns
    assert len(df_pv) == 0

def test_pv_cashflows_non_numeric_discount_rate(sample_dataframe):
    with pytest.raises(TypeError):
        pv_cashflows(sample_dataframe.copy(), 'abc')
