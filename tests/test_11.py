import pytest
import pandas as pd
from definition_249c127d6ebc4d8aa154d73d626e6889 import temporal_split

@pytest.fixture
def sample_dataframe():
    data = {'col1': range(100), 'col2': range(100, 200)}
    df = pd.DataFrame(data)
    return df

def test_temporal_split_valid_train_size(sample_dataframe):
    train_df, oot_df = temporal_split(sample_dataframe, 0.7)
    assert len(train_df) == 70
    assert len(oot_df) == 30
    assert all(train_df.index < 70)
    assert all(oot_df.index >= 70)

def test_temporal_split_train_size_one(sample_dataframe):
    train_df, oot_df = temporal_split(sample_dataframe, 1.0)
    assert len(train_df) == 100
    assert len(oot_df) == 0

def test_temporal_split_train_size_zero(sample_dataframe):
    train_df, oot_df = temporal_split(sample_dataframe, 0.0)
    assert len(train_df) == 0
    assert len(oot_df) == 100

def test_temporal_split_empty_dataframe():
    df = pd.DataFrame()
    train_df, oot_df = temporal_split(df, 0.5)
    assert len(train_df) == 0
    assert len(oot_df) == 0

def test_temporal_split_invalid_train_size(sample_dataframe):
    with pytest.raises(ValueError):
        temporal_split(sample_dataframe, 1.5)