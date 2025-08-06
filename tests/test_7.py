import pytest
import pandas as pd
from definition_bb9cce45c19848629cb08aa73b9022f6 import assign_grade_group

def test_assign_grade_group_empty():
    """Test with an empty DataFrame."""
    df = pd.DataFrame()
    expected = pd.Series([], dtype='object')
    assert assign_grade_group(df).equals(expected)

def test_assign_grade_group_prime():
    """Test with loan grades A and B."""
    data = {'grade': ['A', 'B', 'A', 'B']}
    df = pd.DataFrame(data)
    expected = pd.Series(['Prime'] * 4)
    result = assign_grade_group(df)
    assert result.tolist() == expected.tolist()

def test_assign_grade_group_subprime():
    """Test with loan grades C to G."""
    data = {'grade': ['C', 'D', 'E', 'F', 'G']}
    df = pd.DataFrame(data)
    expected = pd.Series(['Sub-prime'] * 5)
    result = assign_grade_group(df)
    assert result.tolist() == expected.tolist()

def test_assign_grade_group_mixed():
    """Test with a mix of prime and subprime loan grades."""
    data = {'grade': ['A', 'C', 'B', 'E', 'D']}
    df = pd.DataFrame(data)
    expected = pd.Series(['Prime', 'Sub-prime', 'Prime', 'Sub-prime', 'Sub-prime'])
    result = assign_grade_group(df)
    assert result.tolist() == expected.tolist()

def test_assign_grade_group_invalid_grade():
    """Test with invalid loan grades."""
    data = {'grade': ['H', 'I', 'J']}
    df = pd.DataFrame(data)
    expected = pd.Series(['Other'] * 3)
    result = assign_grade_group(df)
    assert result.tolist() == ['Other', 'Other', 'Other']
