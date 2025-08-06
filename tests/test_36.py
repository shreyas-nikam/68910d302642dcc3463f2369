import pytest
import pandas as pd
from definition_55a0abd0540d4068ae86d06a9cab7abd import export_oot

def test_export_oot_valid_data(tmp_path):
    """Test that the function saves the DataFrame to the specified path."""
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    file_path = tmp_path / "test.parquet"
    export_oot(df, file_path)
    assert file_path.exists()

def test_export_oot_empty_dataframe(tmp_path):
    """Test that the function handles an empty DataFrame gracefully."""
    df = pd.DataFrame()
    file_path = tmp_path / "empty.parquet"
    export_oot(df, file_path)
    assert file_path.exists()

def test_export_oot_invalid_path(tmp_path):
    """Test if an exception is raised if a non-string path is given"""
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    file_path = tmp_path / "test.parquet"
    with pytest.raises(TypeError):
        export_oot(df, 123)

def test_export_oot_large_dataframe(tmp_path):
        """Tests that the function does not raise exception with a large DataFrame"""
        df = pd.DataFrame({'col1': range(100000), 'col2': range(100000, 200000)})
        file_path = tmp_path / "large.parquet"
        export_oot(df, str(file_path)) # Convert to string because pathlib object might not be directly compatible.
        assert file_path.exists()
        
def test_export_oot_different_file_extensions(tmp_path):
    """Test exporting to different file extensions (e.g., csv)."""
    df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    file_path_csv = tmp_path / "test.csv"
    export_oot(df, str(file_path_csv)) # Convert to string because pathlib object might not be directly compatible.
    assert file_path_csv.exists()
