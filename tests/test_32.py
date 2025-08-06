import pytest
import pandas as pd
from definition_638ecddd236b4829a14c421b1da30efd import save_quarterly_snapshots

def test_save_quarterly_snapshots_empty_dataframe(tmp_path):
    """Test saving an empty DataFrame."""
    filename = tmp_path / "empty_snapshots.csv"
    df = pd.DataFrame()
    save_quarterly_snapshots(df, str(filename))
    assert filename.exists()
    assert pd.read_csv(filename).empty

def test_save_quarterly_snapshots_valid_dataframe(tmp_path):
    """Test saving a DataFrame with some data."""
    filename = tmp_path / "valid_snapshots.csv"
    data = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data)
    save_quarterly_snapshots(df, str(filename))
    assert filename.exists()
    loaded_df = pd.read_csv(filename)
    pd.testing.assert_frame_equal(df, loaded_df)

def test_save_quarterly_snapshots_different_data_types(tmp_path):
    """Test saving a DataFrame with different data types."""
    filename = tmp_path / "mixed_snapshots.csv"
    data = {'col1': [1, 2], 'col2': ['a', 'b'], 'col3': [1.1, 2.2]}
    df = pd.DataFrame(data)
    save_quarterly_snapshots(df, str(filename))
    assert filename.exists()
    loaded_df = pd.read_csv(filename)
    pd.testing.assert_frame_equal(df, loaded_df)

def test_save_quarterly_snapshots_file_already_exists(tmp_path):
    """Test saving a DataFrame when the file already exists (should overwrite)."""
    filename = tmp_path / "existing_snapshots.csv"
    filename.write_text("initial content")
    data = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data)
    save_quarterly_snapshots(df, str(filename))
    assert filename.exists()
    loaded_df = pd.read_csv(filename)
    pd.testing.assert_frame_equal(df, loaded_df)

def test_save_quarterly_snapshots_filename_with_spaces(tmp_path):
    """Test saving with a filename containing spaces."""
    filename = tmp_path / "file with spaces.csv"
    data = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data)
    save_quarterly_snapshots(df, str(filename))
    assert filename.exists()
    loaded_df = pd.read_csv(filename)
    pd.testing.assert_frame_equal(df, loaded_df)
