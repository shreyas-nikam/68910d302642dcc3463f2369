import pytest
import pandas as pd
from unittest.mock import patch
from definition_652ea01a5a5441da8c05ddd0a1ef6094 import fetch_fred_series

@pytest.fixture
def mock_fred_data():
    # Mock data for testing
    data = {'UNRATE': [5.0, 5.1, 5.2], 'GDP': [1000, 1010, 1020]}
    index = pd.to_datetime(['2023-01-01', '2023-04-01', '2023-07-01'])
    return pd.DataFrame(data, index=index)

@patch('pandas_datareader.data.DataReader')
def test_fetch_fred_series_success(mock_datareader, mock_fred_data):
    mock_datareader.return_value = mock_fred_data
    
    series = {'UNRATE': 'UNRATE', 'GDP': 'GDP'}
    start = '2023-01-01'
    end = '2023-07-01'
    api_key = 'test_api_key'
    
    result = fetch_fred_series(series, start, end, api_key)
    
    assert isinstance(result, pd.DataFrame)
    assert all(col in result.columns for col in series.keys())
    assert len(result) == len(mock_fred_data)

@patch('pandas_datareader.data.DataReader')
def test_fetch_fred_series_empty_series(mock_datareader):
    series = {}
    start = '2023-01-01'
    end = '2023-07-01'
    api_key = 'test_api_key'

    result = fetch_fred_series(series, start, end, api_key)

    assert isinstance(result, pd.DataFrame)
    assert result.empty

@patch('pandas_datareader.data.DataReader')
def test_fetch_fred_series_invalid_date_range(mock_datareader):
    series = {'UNRATE': 'UNRATE'}
    start = '2023-07-01'
    end = '2023-01-01'
    api_key = 'test_api_key'
    
    # Mock the API call to avoid hitting the actual FRED API during testing
    mock_datareader.side_effect = ValueError("Start date must be before end date")
    with pytest.raises(ValueError, match="Start date must be before end date"):
        fetch_fred_series(series, start, end, api_key)

@patch('pandas_datareader.data.DataReader')
def test_fetch_fred_series_missing_api_key(mock_datareader):

    series = {'UNRATE': 'UNRATE'}
    start = '2023-01-01'
    end = '2023-07-01'
    api_key = None

    # Simulate an exception from fred api when key is missing.
    mock_datareader.side_effect = ValueError("API key is required")
    with pytest.raises(ValueError, match="API key is required"):
        fetch_fred_series(series, start, end, api_key)

@patch('pandas_datareader.data.DataReader')
def test_fetch_fred_series_wrong_series_id(mock_datareader, mock_fred_data):
    mock_datareader.return_value = mock_fred_data
    series = {'INVALID': 'INVALID_SERIES'}
    start = '2023-01-01'
    end = '2023-07-01'
    api_key = 'test_api_key'
    
    # Mock the API call to avoid hitting the actual FRED API during testing
    mock_datareader.side_effect = Exception("Invalid FRED series ID")

    with pytest.raises(Exception, match="Invalid FRED series ID"):
        fetch_fred_series(series, start, end, api_key)
