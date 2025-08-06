import pytest
import pandas as pd
from definition_968e2135273540ab8316d888c865d5c7 import temporal_split

@pytest.fixture
def sample_dataframe():
    data = {'date': pd.to_datetime(['2021-01-01', '2021-06-01', '2022-01-01', '2022-06-01', '2023-01-01']),
            'value': [1, 2, 3, 4, 5]}
    return pd.DataFrame(data)

def test_temporal_split_valid_ranges(sample_dataframe):
    train_span = ('2021-01-01', '2021-12-31')
    val_span = ('2022-01-01', '2022-12-31')
    oot_span = ('2023-01-01', '2023-12-31')
    train_df, val_df, oot_df = temporal_split(sample_dataframe, train_span, val_span, oot_span)
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(oot_df, pd.DataFrame)

def test_temporal_split_empty_dataframe():
    df = pd.DataFrame()
    train_span = ('2021-01-01', '2021-12-31')
    val_span = ('2022-01-01', '2022-12-31')
    oot_span = ('2023-01-01', '2023-12-31')
    train_df, val_df, oot_df = temporal_split(df, train_span, val_span, oot_span)
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(oot_df, pd.DataFrame)

def test_temporal_split_no_overlap(sample_dataframe):
    train_span = ('2020-01-01', '2020-12-31')
    val_span = ('2021-01-01', '2021-12-31')
    oot_span = ('2022-01-01', '2022-12-31')
    train_df, val_df, oot_df = temporal_split(sample_dataframe, train_span, val_span, oot_span)
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(oot_df, pd.DataFrame)

def test_temporal_split_partial_overlap(sample_dataframe):
    train_span = ('2021-01-01', '2022-01-01')
    val_span = ('2021-06-01', '2022-06-01')
    oot_span = ('2023-01-01', '2023-12-31')
    train_df, val_df, oot_df = temporal_split(sample_dataframe, train_span, val_span, oot_span)
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(val_df, pd.DataFrame)
    assert isinstance(oot_df, pd.DataFrame)

def test_temporal_split_invalid_date_format(sample_dataframe):
    train_span = ('2021-01-01', '2021-12-31')
    val_span = ('2022-01-01', '2022-12-31')
    oot_span = ('invalid date', '2023-12-31')
    with pytest.raises(Exception):
        temporal_split(sample_dataframe, train_span, val_span, oot_span)
