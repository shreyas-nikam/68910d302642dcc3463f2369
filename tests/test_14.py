import pytest
import pandas as pd
from definition_5f3de2979338468dbf881a18b05f9884 import build_features

def test_build_features_empty_dataframe():
    df = pd.DataFrame()
    result = build_features(df)
    assert isinstance(result, pd.DataFrame)
    assert result.empty

def test_build_features_with_relevant_columns():
    data = {'loan_amnt': [10000, 20000], 'int_rate': [0.10, 0.12], 'term': [36, 60]}
    df = pd.DataFrame(data)
    result = build_features(df)
    assert isinstance(result, pd.DataFrame)

def test_build_features_missing_columns():
    data = {'loan_amnt': [10000, 20000]}
    df = pd.DataFrame(data)
    result = build_features(df)
    assert isinstance(result, pd.DataFrame)
    #Should still return dataframe without error

def test_build_features_all_nan_values():
    data = {'loan_amnt': [None, None], 'int_rate': [None, None], 'term': [None, None]}
    df = pd.DataFrame(data)
    result = build_features(df)
    assert isinstance(result, pd.DataFrame)

def test_build_features_mixed_data_types():
    data = {'loan_amnt': [10000, "20000"], 'int_rate': [0.10, "0.12"], 'term': [36, 60]}
    df = pd.DataFrame(data)
    result = build_features(df)
    assert isinstance(result, pd.DataFrame)

