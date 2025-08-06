import pytest
import pandas as pd
from definition_0b0b0e49c6374f55abcf8b875ab0b19c import save_parquet

def test_save_parquet_success(tmp_path):
    """Test saving a DataFrame to Parquet successfully."""
    df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    file_path = tmp_path / "test.parquet"
    save_parquet(df, str(file_path))
    assert file_path.exists()

def test_save_parquet_empty_dataframe(tmp_path):
    """Test saving an empty DataFrame to Parquet."""
    df = pd.DataFrame()
    file_path = tmp_path / "empty.parquet"
    save_parquet(df, str(file_path))
    assert file_path.exists()

def test_save_parquet_invalid_path(tmp_path):
    """Test saving to an invalid path."""
    df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    file_path = 123  # Not a string
    with pytest.raises(TypeError):  # Expecting some kind of type error due to file_path
        save_parquet(df, file_path)

def test_save_parquet_large_dataframe(tmp_path):
    """Test saving a large DataFrame to Parquet."""
    data = {'col1': list(range(1000)), 'col2': ['a'] * 1000}
    df = pd.DataFrame(data)
    file_path = tmp_path / "large.parquet"
    save_parquet(df, str(file_path))
    assert file_path.exists()

def test_save_parquet_non_dataframe_input(tmp_path):
    """Test saving a non-DataFrame object."""
    not_a_df = [1, 2, 3]
    file_path = tmp_path / "wrong_input.parquet"

    with pytest.raises(AttributeError): #or TypeError depending on implementation
        save_parquet(not_a_df, str(file_path))
