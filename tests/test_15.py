import pytest
import pandas as pd
from definition_5a61c14bea4848949831b98985e1099f import aggregate_lgd_by_cohort

@pytest.fixture
def sample_df():
    data = {'origination_date': pd.to_datetime(['2020-01-01', '2020-01-01', '2020-02-01', '2020-02-01', '2020-03-01']),
            'lgd': [0.1, 0.2, 0.3, 0.4, 0.5]}
    return pd.DataFrame(data)

def test_aggregate_lgd_by_cohort_empty_df():
    df = pd.DataFrame()
    result = aggregate_lgd_by_cohort(df)
    assert result.empty

def test_aggregate_lgd_by_cohort_basic_aggregation(sample_df):
    result = aggregate_lgd_by_cohort(sample_df)
    expected_data = {'origination_date': pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01']),
                     'lgd': [0.15, 0.35, 0.5]}
    expected = pd.DataFrame(expected_data).set_index('origination_date')
    pd.testing.assert_frame_equal(result, expected)

def test_aggregate_lgd_by_cohort_with_other_columns(sample_df):
    sample_df['loan_amount'] = [1000, 2000, 1500, 2500, 3000]
    result = aggregate_lgd_by_cohort(sample_df)
    expected_data = {'origination_date': pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01']),
                     'lgd': [0.15, 0.35, 0.5]}
    expected = pd.DataFrame(expected_data).set_index('origination_date')
    pd.testing.assert_frame_equal(result, expected)

def test_aggregate_lgd_by_cohort_different_date_formats():
    data = {'origination_date': ['2020-01-01', '2020-01-01', '2020-02-01', '2020-02-01', '2020-03-01'],
            'lgd': [0.1, 0.2, 0.3, 0.4, 0.5]}
    df = pd.DataFrame(data)
    df['origination_date'] = pd.to_datetime(df['origination_date'])
    result = aggregate_lgd_by_cohort(df)
    expected_data = {'origination_date': pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01']),
                     'lgd': [0.15, 0.35, 0.5]}
    expected = pd.DataFrame(expected_data).set_index('origination_date')
    pd.testing.assert_frame_equal(result, expected)

def test_aggregate_lgd_by_cohort_missing_lgd_values():
    data = {'origination_date': pd.to_datetime(['2020-01-01', '2020-01-01', '2020-02-01', '2020-02-01', '2020-03-01']),
            'lgd': [0.1, None, 0.3, 0.4, 0.5]}
    df = pd.DataFrame(data)
    result = aggregate_lgd_by_cohort(df)
    expected_data = {'origination_date': pd.to_datetime(['2020-01-01', '2020-02-01', '2020-03-01']),
                     'lgd': [0.1, 0.35, 0.5]}
    expected = pd.DataFrame(expected_data).set_index('origination_date')
    pd.testing.assert_frame_equal(result, expected)
