import pytest
import pandas as pd
from definition_ed34be258a3f49cb838d92772a341927 import align_macro_with_cohorts

@pytest.fixture
def sample_lgd_data():
    return pd.DataFrame({
        'cohort': ['2020-Q1', '2020-Q2', '2020-Q3'],
        'lgd': [0.1, 0.2, 0.3]
    })

@pytest.fixture
def sample_macro_data():
    return pd.DataFrame({
        'date': ['2020-03-31', '2020-06-30', '2020-09-30'],
        'gdp_growth': [0.01, 0.02, 0.03]
    })

def test_align_macro_with_cohorts_empty_input():
    lgd_data = pd.DataFrame()
    macro_data = pd.DataFrame()
    result = align_macro_with_cohorts(lgd_data, macro_data)
    assert isinstance(result, pd.DataFrame)
    assert result.empty

def test_align_macro_with_cohorts_basic_alignment(sample_lgd_data, sample_macro_data):
    result = align_macro_with_cohorts(sample_lgd_data, sample_macro_data)
    assert isinstance(result, pd.DataFrame)
    assert 'gdp_growth' in result.columns
    assert len(result) == len(sample_lgd_data)
    

def test_align_macro_with_cohorts_missing_macro_data(sample_lgd_data):
    macro_data = pd.DataFrame({
        'date': ['2020-03-31'],
        'gdp_growth': [0.01]
    })
    result = align_macro_with_cohorts(sample_lgd_data, macro_data)
    assert isinstance(result, pd.DataFrame)
    assert 'gdp_growth' in result.columns

def test_align_macro_with_cohorts_different_date_formats(sample_lgd_data):
    macro_data = pd.DataFrame({
        'date': ['3/31/2020', '6/30/2020', '9/30/2020'],
        'gdp_growth': [0.01, 0.02, 0.03]
    })
    try:
      result = align_macro_with_cohorts(sample_lgd_data, macro_data)
    except Exception as e:
      assert isinstance(e, KeyError)

def test_align_macro_with_cohorts_no_cohort_column(sample_macro_data):
    lgd_data = pd.DataFrame({
        'date': ['2020-Q1', '2020-Q2', '2020-Q3'],
        'lgd': [0.1, 0.2, 0.3]
    })
    with pytest.raises(KeyError):  # Assuming a KeyError if cohort column is missing
      align_macro_with_cohorts(lgd_data, sample_macro_data)
