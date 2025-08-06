import pytest
import pandas as pd
from definition_6d9704dc980d4031ad4acd4b19a81550 import validate_required_columns

def test_validate_required_columns_success():
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    cols = ['col1', 'col2']
    validate_required_columns(df, cols)  # Should not raise an exception

def test_validate_required_columns_missing_column():
    df = pd.DataFrame({'col1': [1, 2]})
    cols = ['col1', 'col2']
    with pytest.raises(ValueError) as excinfo:
        validate_required_columns(df, cols)
    assert "Missing required columns: ['col2']" in str(excinfo.value)

def test_validate_required_columns_empty_dataframe():
    df = pd.DataFrame()
    cols = ['col1']
    with pytest.raises(ValueError) as excinfo:
        validate_required_columns(df, cols)
    assert "Missing required columns: ['col1']" in str(excinfo.value)

def test_validate_required_columns_empty_cols_list():
     df = pd.DataFrame({'col1': [1, 2]})
     cols: list[str] = []
     validate_required_columns(df, cols) #Should not raise an exception

def test_validate_required_columns_duplicate_columns():
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    cols = ['col1', 'col1', 'col2']
    with pytest.raises(ValueError) as excinfo:
        validate_required_columns(df, cols)
    assert "Duplicate columns provided in required columns list." in str(excinfo.value)
