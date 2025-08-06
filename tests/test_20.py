import pytest
import pandas as pd
from definition_b4abf398c37d41f89b6e75b83f410f9f import aggregate_lgd_by_cohort

@pytest.fixture
def sample_df():
    data = {'default_quarter': ['2020Q1', '2020Q1', '2020Q2', '2020Q2', '2020Q3'],
            'LGD_realized': [0.1, 0.2, 0.3, 0.4, 0.5]}
    return pd.DataFrame(data)

def test_aggregate_lgd_by_cohort_empty_df():
    df = pd.DataFrame({'default_quarter': [], 'LGD_realized': []})
    result = aggregate_lgd_by_cohort(df)
    assert result.empty

def test_aggregate_lgd_by_cohort_standard(sample_df):
    result = aggregate_lgd_by_cohort(sample_df)
    expected_data = {'default_quarter': ['2020Q1', '2020Q2', '2020Q3'],
                     'LGD_realized': [0.15, 0.35, 0.5]}
    expected = pd.DataFrame(expected_data)
    expected = expected.set_index('default_quarter')

    pd.testing.assert_frame_equal(result, expected)

def test_aggregate_lgd_by_cohort_missing_lgd(sample_df):
    sample_df['LGD_realized'] = None
    result = aggregate_lgd_by_cohort(sample_df)

    expected_data = {'default_quarter': ['2020Q1', '2020Q2', '2020Q3'],
                     'LGD_realized': [None, None, None]}
    expected = pd.DataFrame(expected_data)
    expected = expected.set_index('default_quarter')
    pd.testing.assert_frame_equal(result, expected)

def test_aggregate_lgd_by_cohort_single_cohort():
    data = {'default_quarter': ['2020Q1', '2020Q1', '2020Q1'],
            'LGD_realized': [0.1, 0.2, 0.3]}
    df = pd.DataFrame(data)
    result = aggregate_lgd_by_cohort(df)
    expected_data = {'default_quarter': ['2020Q1'],
                     'LGD_realized': [0.2]}
    expected = pd.DataFrame(expected_data)
    expected = expected.set_index('default_quarter')
    pd.testing.assert_frame_equal(result, expected)
