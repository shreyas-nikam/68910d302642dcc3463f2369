import pytest
import pandas as pd
from definition_eae95140b72240588dff5004b738c20f import add_default_quarter

@pytest.fixture
def sample_df():
    data = {'default_date': ['2019-01-15', '2019-04-20', '2019-07-10', '2019-10-05', '2020-01-22']}
    return pd.DataFrame(data)

def test_add_default_quarter_basic(sample_df):
    df = add_default_quarter(sample_df.copy())
    assert 'default_quarter' in df.columns
    assert df['default_quarter'].tolist() == ['2019Q1', '2019Q2', '2019Q3', '2019Q4', '2020Q1']

def test_add_default_quarter_empty_df():
    df = pd.DataFrame()
    df = add_default_quarter(df.copy())
    assert 'default_quarter' in df.columns
    assert len(df) == 0

def test_add_default_quarter_incorrect_date_format(sample_df):
    sample_df['default_date'] = ['15-01-2019', '20-04-2019', '10-07-2019', '05-10-2019', '22-01-2020']
    with pytest.raises(ValueError):
        add_default_quarter(sample_df.copy())

def test_add_default_quarter_missing_default_date():
    df = pd.DataFrame({'other_col': [1, 2, 3]})
    with pytest.raises(KeyError):
        add_default_quarter(df.copy())

def test_add_default_quarter_non_string_dates():
    df = pd.DataFrame({'default_date': [1, 2, 3, 4, 5]})
    with pytest.raises(TypeError):
         add_default_quarter(df.copy())
