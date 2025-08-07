import pytest
from definition_eafb64563471476286e4de5db5b9b55a import fetch_lendingclub_date
import pandas as pd
from unittest.mock import patch
import io

@pytest.fixture
def mock_csv_data():
    # Minimal CSV data to avoid external dependencies for testing
    csv_data = """loan_amnt,int_rate,grade
1000,0.10,A
2000,0.12,B
3000,0.14,C"""
    return csv_data


@patch('pandas.read_csv')
@patch('zipfile.ZipFile')
@patch('requests.get')
def test_fetch_lendingclub_date_success(mock_get, mock_zipfile, mock_read_csv, mock_csv_data):
    """Test successful fetching and loading of data."""
    mock_get.return_value.content = b'zip content'
    mock_zipfile.return_value.__enter__.return_value.namelist.return_value = ['LoanStats_2018Q4.csv']
    mock_zipfile.return_value.__enter__.return_value.open.return_value = io.StringIO(mock_csv_data)
    mock_read_csv.return_value = pd.DataFrame({'loan_amnt': [1000, 2000, 3000], 'int_rate': [0.10, 0.12, 0.14], 'grade': ['A', 'B', 'C']})

    df = fetch_lendingclub_date()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(df) == 3


@patch('requests.get')
def test_fetch_lendingclub_date_connection_error(mock_get):
    """Test handling of connection errors."""
    mock_get.side_effect = Exception("Connection error")
    with pytest.raises(Exception, match="Connection error"):
        fetch_lendingclub_date()



@patch('pandas.read_csv')
@patch('zipfile.ZipFile')
@patch('requests.get')
def test_fetch_lendingclub_date_empty_zip(mock_get, mock_zipfile, mock_read_csv):
    """Test handling of empty zip file."""
    mock_get.return_value.content = b'zip content'
    mock_zipfile.return_value.__enter__.return_value.namelist.return_value = [] # Simulate empty zip

    with pytest.raises(Exception, match="No CSV file found in the zip archive."):
        fetch_lendingclub_date()


@patch('pandas.read_csv')
@patch('zipfile.ZipFile')
@patch('requests.get')
def test_fetch_lendingclub_date_invalid_csv(mock_get, mock_zipfile, mock_read_csv, mock_csv_data):
    """Test handling of invalid CSV file (e.g., missing columns)."""
    mock_get.return_value.content = b'zip content'
    mock_zipfile.return_value.__enter__.return_value.namelist.return_value = ['LoanStats_2018Q4.csv']
    mock_zipfile.return_value.__enter__.return_value.open.return_value = io.StringIO(mock_csv_data)
    mock_read_csv.side_effect = pd.errors.ParserError("CSV parsing error")

    with pytest.raises(pd.errors.ParserError, match="CSV parsing error"):
        fetch_lendingclub_date()

