import pytest
import pandas as pd
from definition_3fc93de26eb34f77807189e89bfe5d26 import read_lendingclub

def test_read_lendingclub_csv(tmp_path):
    # Create a dummy CSV file
    file_path = tmp_path / "test.csv"
    file_path.write_text("loan_amnt,int_rate\n1000,0.10\n2000,0.12")

    df = read_lendingclub(file_path)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ["loan_amnt", "int_rate"]
    assert df["loan_amnt"].tolist() == [1000, 2000]


def test_read_lendingclub_parquet(tmp_path):
    # Create a dummy DataFrame and save it as Parquet
    data = {'loan_amnt': [1000, 2000], 'int_rate': [0.10, 0.12]}
    df = pd.DataFrame(data)
    file_path = tmp_path / "test.parquet"
    df.to_parquet(file_path)

    df_loaded = read_lendingclub(file_path)
    assert isinstance(df_loaded, pd.DataFrame)
    assert df_loaded.shape == (2, 2)
    assert list(df_loaded.columns) == ["loan_amnt", "int_rate"]
    assert df_loaded["loan_amnt"].tolist() == [1000, 2000]

def test_read_lendingclub_empty_file(tmp_path):
    # Create an empty file
    file_path = tmp_path / "empty.csv"
    file_path.write_text("")

    df = read_lendingclub(file_path)
    assert isinstance(df, pd.DataFrame)
    assert df.empty

def test_read_lendingclub_invalid_file_path():
    # Test with an invalid file path
    with pytest.raises(FileNotFoundError):
        read_lendingclub("nonexistent_file.csv")

def test_read_lendingclub_unsupported_file_type(tmp_path):
    #Create a dummy txt file.
    file_path = tmp_path / "test.txt"
    file_path.write_text("loan_amnt,int_rate\n1000,0.10\n2000,0.12")
    with pytest.raises(ValueError, match="Unsupported file format"):
        read_lendingclub(file_path)

