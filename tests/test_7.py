import pytest
import pandas as pd
from definition_e0a3548237ff4a90b921c1856965dfb0 import assign_grade_group

@pytest.fixture
def sample_dataframe():
    data = {'grade': ['A', 'B', 'C', 'D', 'E', 'F', 'G']}
    return pd.DataFrame(data)

def test_assign_grade_group_prime(sample_dataframe):
    df = sample_dataframe.copy()
    result_df = assign_grade_group(df)
    assert 'grade_group' in result_df.columns
    assert result_df['grade_group'][0] == 'Prime'
    assert result_df['grade_group'][1] == 'Prime'

def test_assign_grade_group_subprime(sample_dataframe):
    df = sample_dataframe.copy()
    result_df = assign_grade_group(df)
    assert 'grade_group' in result_df.columns
    assert result_df['grade_group'][2] == 'Sub-prime'
    assert result_df['grade_group'][6] == 'Sub-prime'

def test_assign_grade_group_empty_dataframe():
    df = pd.DataFrame()
    result_df = assign_grade_group(df)
    assert 'grade_group' in result_df.columns
    assert len(result_df) == 0

def test_assign_grade_group_mixed_grades():
    data = {'grade': ['A', 'C', 'B', 'F']}
    df = pd.DataFrame(data)
    result_df = assign_grade_group(df)
    assert result_df['grade_group'][0] == 'Prime'
    assert result_df['grade_group'][1] == 'Sub-prime'
    assert result_df['grade_group'][2] == 'Prime'
    assert result_df['grade_group'][3] == 'Sub-prime'

def test_assign_grade_group_missing_grade_column():
    df = pd.DataFrame({'loan_amount': [1000, 2000]})
    with pytest.raises(KeyError) as excinfo:
        assign_grade_group(df)
    assert "grade" in str(excinfo.value)
