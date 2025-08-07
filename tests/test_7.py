import pytest
import pandas as pd
from definition_6d06ad7b462642ffbeeef2b5b08f48c9 import assign_grade_group

@pytest.fixture
def sample_dataframe():
    data = {'grade': ['A', 'B', 'C', 'D', 'E', 'F', 'G']}
    return pd.DataFrame(data)

def test_assign_grade_group_prime(sample_dataframe):
    df = assign_grade_group(sample_dataframe)
    assert 'grade_group' in df.columns
    assert df['grade_group'].iloc[0] == 'Prime'
    assert df['grade_group'].iloc[1] == 'Prime'

def test_assign_grade_group_subprime(sample_dataframe):
    df = assign_grade_group(sample_dataframe)
    assert 'grade_group' in df.columns
    assert df['grade_group'].iloc[2] == 'Sub-prime'
    assert df['grade_group'].iloc[3] == 'Sub-prime'
    assert df['grade_group'].iloc[4] == 'Sub-prime'
    assert df['grade_group'].iloc[5] == 'Sub-prime'
    assert df['grade_group'].iloc[6] == 'Sub-prime'

def test_assign_grade_group_empty_dataframe():
    df = pd.DataFrame({'grade': []})
    df = assign_grade_group(df)
    assert 'grade_group' in df.columns
    assert len(df) == 0

def test_assign_grade_group_mixed_grades(sample_dataframe):
    mixed_data = {'grade': ['A', 'C', 'B', 'F']}
    df = pd.DataFrame(mixed_data)
    df = assign_grade_group(df)
    assert 'grade_group' in df.columns
    assert df['grade_group'].iloc[0] == 'Prime'
    assert df['grade_group'].iloc[1] == 'Sub-prime'
    assert df['grade_group'].iloc[2] == 'Prime'
    assert df['grade_group'].iloc[3] == 'Sub-prime'

def test_assign_grade_group_invalid_grade():
    data = {'grade': ['H']}
    df = pd.DataFrame(data)
    df = assign_grade_group(df)
    assert 'grade_group' in df.columns
    assert df['grade_group'].iloc[0] == 'Unknown'
