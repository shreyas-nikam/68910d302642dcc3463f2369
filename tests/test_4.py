import pytest
import pandas as pd
from definition_83e12205f83a4734b644d68a1f401c3a import compute_ead

def test_compute_ead_empty_dataframe():
    df = pd.DataFrame()
    result = compute_ead(df)
    assert isinstance(result, pd.DataFrame)
    assert result.empty

def test_compute_ead_no_ead_column():
    df = pd.DataFrame({'loan_amnt': [1000, 2000], 'int_rate': [0.1, 0.15]})
    with pytest.raises(KeyError):
        compute_ead(df)

def test_compute_ead_valid_dataframe():
    df = pd.DataFrame({'loan_amnt': [1000, 2000], 'int_rate': [0.1, 0.15], 'funded_amnt': [1000,2000]})
    df['funded_amnt']=df['loan_amnt']
    df['total_pymnt']=df['loan_amnt']
    result = compute_ead(df.copy()) #added .copy to deal with SettingWithCopyWarning as the original DF is updated
    assert isinstance(result, pd.DataFrame)
    # Basic check, more specific checks would require a more complete implementation
    assert len(result) == len(df)

def test_compute_ead_negative_values():
    df = pd.DataFrame({'loan_amnt': [-1000, 2000], 'int_rate': [0.1, 0.15], 'funded_amnt': [-1000,2000]})
    df['funded_amnt']=df['loan_amnt']
    df['total_pymnt']=df['loan_amnt']
    result = compute_ead(df.copy())#added .copy to deal with SettingWithCopyWarning as the original DF is updated
    assert isinstance(result, pd.DataFrame)
    # Basic check, more specific checks would require a more complete implementation
    assert len(result) == len(df)

def test_compute_ead_non_numeric_values():
    df = pd.DataFrame({'loan_amnt': ['abc', 'def'], 'int_rate': [0.1, 0.15], 'funded_amnt': ['abc','def']})
    df['funded_amnt']=df['loan_amnt']
    df['total_pymnt']=df['loan_amnt']
    with pytest.raises(TypeError):
        compute_ead(df.copy())#added .copy to deal with SettingWithCopyWarning as the original DF is updated
