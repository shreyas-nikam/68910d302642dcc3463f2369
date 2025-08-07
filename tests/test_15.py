import pytest
import pandas as pd
from definition_2fca494db7c6407bac814581ec7166e8 import aggregate_lgd_by_cohort

@pytest.fixture
def sample_dataframe():
    data = {'cohort': ['2023-Q1', '2023-Q1', '2023-Q2', '2023-Q2', '2023-Q3'],
            'LGD': [0.1, 0.2, 0.3, 0.4, 0.5]}
    return pd.DataFrame(data)

def test_aggregate_lgd_by_cohort_empty_df():
    df = pd.DataFrame()
    result = aggregate_lgd_by_cohort(df)
    assert result.empty

def test_aggregate_lgd_by_cohort_single_cohort(sample_dataframe):
    df = sample_dataframe[sample_dataframe['cohort'] == '2023-Q1']
    result = aggregate_lgd_by_cohort(df)
    assert not result.empty
    assert 'cohort' in result.columns
    assert 'mean_LGD' in result.columns
    assert result['mean_LGD'].iloc[0] == 0.15

def test_aggregate_lgd_by_cohort_multiple_cohorts(sample_dataframe):
    result = aggregate_lgd_by_cohort(sample_dataframe)
    assert not result.empty
    assert len(result) == 3
    assert 'cohort' in result.columns
    assert 'mean_LGD' in result.columns
    assert result['mean_LGD'].iloc[0] == 0.15 #mean of cohort 2023-Q1

def test_aggregate_lgd_by_cohort_missing_lgd_values(sample_dataframe):
    df = sample_dataframe.copy()
    df.loc[0, 'LGD'] = None
    result = aggregate_lgd_by_cohort(df)
    assert not result.empty
    assert result['mean_LGD'].iloc[0] == 0.2

def test_aggregate_lgd_by_cohort_mixed_data_types():
    data = {'cohort': ['2023-Q1', '2023-Q1', '2023-Q2'],
            'LGD': [0.1, '0.2', 0.3]}
    df = pd.DataFrame(data)
    result = aggregate_lgd_by_cohort(df)
    assert not result.empty
    assert len(result) == 2
    assert result['mean_LGD'].iloc[0] == 0.15
