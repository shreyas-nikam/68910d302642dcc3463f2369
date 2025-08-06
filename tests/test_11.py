import pytest
import pandas as pd
from definition_0cdd64eb07fd4ba2b8718d1fb0a16446 import temporal_split

@pytest.fixture
def sample_dataframe():
    data = {'loan_amnt': [1000, 2000, 3000, 4000, 5000],
            'int_rate': [0.10, 0.12, 0.14, 0.16, 0.18],
            'grade': ['A', 'B', 'C', 'D', 'E']}
    return pd.DataFrame(data)

def test_temporal_split_valid_train_size(sample_dataframe):
    train_size = 0.6
    train_df, test_df = temporal_split(sample_dataframe, train_size)
    assert len(train_df) == 3
    assert len(test_df) == 2

def test_temporal_split_train_size_one(sample_dataframe):
    train_size = 1.0
    train_df, test_df = temporal_split(sample_dataframe, train_size)
    assert len(train_df) == 5
    assert len(test_df) == 0

def test_temporal_split_train_size_zero(sample_dataframe):
    train_size = 0.0
    train_df, test_df = temporal_split(sample_dataframe, train_size)
    assert len(train_df) == 0
    assert len(test_df) == 5

def test_temporal_split_invalid_train_size(sample_dataframe):
    with pytest.raises(ValueError):
        temporal_split(sample_dataframe, 1.5)

def test_temporal_split_empty_dataframe():
    df = pd.DataFrame()
    train_size = 0.5
    train_df, test_df = temporal_split(df, train_size)
    assert len(train_df) == 0
    assert len(test_df) == 0
