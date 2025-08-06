import pytest
import pandas as pd
from definition_bb1b3709107b4132ba5c96df5f700bbf import build_features

def test_build_features_empty_df():
    df = pd.DataFrame()
    result = build_features(df)
    assert isinstance(result, pd.DataFrame)

def test_build_features_with_expected_columns():
    data = {'loan_amnt': [10000], 'int_rate': [0.10], 'installment': [300]}
    df = pd.DataFrame(data)
    result = build_features(df)
    assert isinstance(result, pd.DataFrame)

def test_build_features_nan_values():
    data = {'loan_amnt': [float('nan')], 'int_rate': [0.10], 'installment': [300]}
    df = pd.DataFrame(data)
    result = build_features(df)
    assert isinstance(result, pd.DataFrame)

def test_build_features_categorical_data():
    data = {'grade': ['A'], 'term': ['36 months']}
    df = pd.DataFrame(data)
    result = build_features(df)
    assert isinstance(result, pd.DataFrame)
    
def test_build_features_large_dataframe():
    data = {'loan_amnt': [10000] * 100, 'int_rate': [0.10] * 100, 'installment': [300] * 100}
    df = pd.DataFrame(data)
    result = build_features(df)
    assert isinstance(result, pd.DataFrame)