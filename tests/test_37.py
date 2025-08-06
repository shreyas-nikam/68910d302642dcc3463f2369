import pytest
import pandas as pd
import os
from definition_1b9ae2a3af7f4b1e88208b8ceeae9032 import save_quarterly_snapshots

def test_save_quarterly_snapshots_empty_df(tmpdir):
    df = pd.DataFrame()
    dirpath = str(tmpdir)
    save_quarterly_snapshots(df, dirpath)
    assert len(os.listdir(dirpath)) == 0

def test_save_quarterly_snapshots_single_quarter(tmpdir):
    df = pd.DataFrame({'quarter': ['2023Q1'], 'loan_amount': [1000]})
    dirpath = str(tmpdir)
    save_quarterly_snapshots(df, dirpath)
    assert len(os.listdir(dirpath)) == 1
    filepath = os.path.join(dirpath, 'snap_2023Q1.csv')
    assert os.path.exists(filepath)
    saved_df = pd.read_csv(filepath)
    pd.testing.assert_frame_equal(saved_df, df)

def test_save_quarterly_snapshots_multiple_quarters(tmpdir):
    df = pd.DataFrame({'quarter': ['2023Q1', '2023Q2', '2023Q1'], 'loan_amount': [1000, 2000, 1500]})
    dirpath = str(tmpdir)
    save_quarterly_snapshots(df, dirpath)
    assert len(os.listdir(dirpath)) == 2
    filepath1 = os.path.join(dirpath, 'snap_2023Q1.csv')
    filepath2 = os.path.join(dirpath, 'snap_2023Q2.csv')
    assert os.path.exists(filepath1)
    assert os.path.exists(filepath2)
    saved_df1 = pd.read_csv(filepath1)
    saved_df2 = pd.read_csv(filepath2)

    expected_df1 = df[df['quarter'] == '2023Q1']
    expected_df2 = df[df['quarter'] == '2023Q2']
    pd.testing.assert_frame_equal(saved_df1, expected_df1)
    pd.testing.assert_frame_equal(saved_df2, expected_df2)

def test_save_quarterly_snapshots_invalid_dirpath(tmpdir):
    df = pd.DataFrame({'quarter': ['2023Q1'], 'loan_amount': [1000]})
    dirpath = 123  # Invalid directory path
    with pytest.raises(TypeError):
        save_quarterly_snapshots(df, dirpath)

def test_save_quarterly_snapshots_missing_quarter_column(tmpdir):
    df = pd.DataFrame({'loan_amount': [1000]})
    dirpath = str(tmpdir)
    with pytest.raises(KeyError) as excinfo:
        save_quarterly_snapshots(df, dirpath)
    assert "quarter" in str(excinfo.value)
