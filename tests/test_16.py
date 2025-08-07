import pytest
import pandas as pd
from definition_5ccd3a91e33f4af8be756def28a69016 import align_macro_with_cohorts

@pytest.fixture
def sample_lgd_cohorts():
    return pd.DataFrame({
        'cohort_start_date': pd.to_datetime(['2020-01-01', '2020-04-01', '2020-07-01']),
        'LGD': [0.1, 0.2, 0.3]
    })

@pytest.fixture
def sample_macro_data():
    return pd.DataFrame({
        'date': pd.to_datetime(['2019-12-31', '2020-03-31', '2020-06-30', '2020-09-30']),
        'unemployment_rate': [4.0, 5.0, 6.0, 7.0]
    })

def test_align_macro_with_cohorts_empty_input(sample_lgd_cohorts, sample_macro_data):
    empty_df = pd.DataFrame()
    result = align_macro_with_cohorts(empty_df, sample_macro_data)
    assert result.empty

def test_align_macro_with_cohorts_basic_alignment(sample_lgd_cohorts, sample_macro_data):
    result = align_macro_with_cohorts(sample_lgd_cohorts, sample_macro_data)
    assert isinstance(result, pd.DataFrame)
    assert 'unemployment_rate' in result.columns

def test_align_macro_with_cohorts_no_macro_data(sample_lgd_cohorts):
    empty_macro = pd.DataFrame()
    result = align_macro_with_cohorts(sample_lgd_cohorts, empty_macro)
    assert isinstance(result, pd.DataFrame)
    assert result.equals(sample_lgd_cohorts)

def test_align_macro_with_cohorts_date_alignment(sample_lgd_cohorts, sample_macro_data):
    result = align_macro_with_cohorts(sample_lgd_cohorts, sample_macro_data)
    
    expected_unemployment = [5.0, 6.0, 7.0]
    assert list(result['unemployment_rate']) == expected_unemployment

def test_align_macro_with_cohorts_different_date_format(sample_lgd_cohorts):
    macro_data = pd.DataFrame({
        'date': ['2019-12-31', '2020-03-31', '2020-06-30', '2020-09-30'],
        'unemployment_rate': [4.0, 5.0, 6.0, 7.0]
    })
    result = align_macro_with_cohorts(sample_lgd_cohorts, macro_data)
    assert isinstance(result, pd.DataFrame)
    assert 'unemployment_rate' in result.columns

