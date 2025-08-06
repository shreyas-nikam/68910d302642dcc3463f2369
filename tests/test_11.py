import pytest
import pandas as pd
from definition_73713d9c127b44dca0934eb216ad4bde import temporal_split

@pytest.fixture
def sample_dataframe():
    # Create a sample DataFrame for testing
    data = {'issue_d': pd.to_datetime(['2017-01-15', '2017-06-20', '2018-03-10', '2019-09-05', '2020-02-28'])}
    return pd.DataFrame(data)

def test_temporal_split_valid(sample_dataframe):
    # Test that the function returns two dataframes
    train_df, oot_df = temporal_split(sample_dataframe)
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(oot_df, pd.DataFrame)

def test_temporal_split_oot_year(sample_dataframe):
    # Test that OOT dataframe contains data from 2019 onwards
    train_df, oot_df = temporal_split(sample_dataframe)
    assert all(oot_df['issue_d'].dt.year >= 2019)

def test_temporal_split_train_year(sample_dataframe):
    # Test that training dataframe contains data before 2019
    train_df, oot_df = temporal_split(sample_dataframe)
    assert all(train_df['issue_d'].dt.year < 2019)

def test_temporal_split_empty_input():
    # Test with an empty DataFrame
    empty_df = pd.DataFrame()
    train_df, oot_df = temporal_split(empty_df)
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(oot_df, pd.DataFrame)
    assert train_df.empty
    assert oot_df.empty

def test_temporal_split_no_oot_data():
   # Test that function works correctly when there's no data after the OOT split year
    data = {'issue_d': pd.to_datetime(['2016-01-15', '2017-06-20', '2018-03-10'])}
    df = pd.DataFrame(data)
    train_df, oot_df = temporal_split(df)
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(oot_df, pd.DataFrame)
    assert not train_df.empty
    assert oot_df.empty
    assert all(train_df['issue_d'].dt.year < 2019)

