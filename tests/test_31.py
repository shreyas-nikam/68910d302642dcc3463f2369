import pytest
import pandas as pd
from definition_ba0bdc53ebfa4905b339ba3da1599259 import export_oot

def test_export_oot_valid_data(tmp_path):
    # Test with valid DataFrame and filename
    data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    filename = tmp_path / "test.parquet"
    export_oot(data, str(filename))
    assert filename.exists()

def test_export_oot_empty_data(tmp_path):
    # Test with empty DataFrame
    data = pd.DataFrame()
    filename = tmp_path / "test_empty.parquet"
    export_oot(data, str(filename))
    assert filename.exists()

def test_export_oot_invalid_filename(tmp_path):
     # Test with invalid filename
    data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    filename = tmp_path / "test.txt"

    with pytest.raises(ValueError):  # Expect an error depending on implementation
        export_oot(data, str(filename))

def test_export_oot_filename_none(tmp_path):
    data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    with pytest.raises(TypeError):
        export_oot(data, None)

def test_export_oot_parquet_format(tmp_path):
    data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
    filename = tmp_path / "test.parquet"
    export_oot(data, str(filename))
    df = pd.read_parquet(filename)
    pd.testing.assert_frame_equal(df, data)
