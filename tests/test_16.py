import pytest
import pandas as pd
from definition_6b3f4597671d46dcaae4bd8e48c4bb1e import aggregate_lgd_by_cohort

def test_aggregate_lgd_by_cohort_empty_dataframe(monkeypatch):
    # Test case 1: Empty DataFrame
    def mock_read_csv(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(pd, 'read_csv', mock_read_csv)

    result = aggregate_lgd_by_cohort()
    assert isinstance(result, pd.DataFrame)
    assert result.empty

def test_aggregate_lgd_by_cohort_no_default_quarter(monkeypatch):
    # Test case 2: DataFrame without 'default_quarter' column
    def mock_read_csv(*args, **kwargs):
        return pd.DataFrame({'loan_id': [1, 2], 'lgd': [0.1, 0.2]})

    monkeypatch.setattr(pd, 'read_csv', mock_read_csv)

    with pytest.raises(KeyError) as excinfo:
        aggregate_lgd_by_cohort()
    assert "default_quarter" in str(excinfo.value)

def test_aggregate_lgd_by_cohort_basic_aggregation(monkeypatch):
    # Test case 3: Basic aggregation functionality
    def mock_read_csv(*args, **kwargs):
        return pd.DataFrame({
            'default_quarter': ['2023-Q1', '2023-Q1', '2023-Q2', '2023-Q2'],
            'lgd': [0.1, 0.2, 0.3, 0.4]
        })

    monkeypatch.setattr(pd, 'read_csv', mock_read_csv)

    result = aggregate_lgd_by_cohort()
    assert isinstance(result, pd.DataFrame)
    assert 'default_quarter' in result.columns
    assert 'mean_lgd' in result.columns
    assert len(result) == 2  # Two unique quarters

    # Check values for 2023-Q1
    q1_data = result[result['default_quarter'] == '2023-Q1']['mean_lgd'].values[0]
    assert q1_data == 0.15

    # Check values for 2023-Q2
    q2_data = result[result['default_quarter'] == '2023-Q2']['mean_lgd'].values[0]
    assert q2_data == 0.35

def test_aggregate_lgd_by_cohort_missing_lgd_values(monkeypatch):
    # Test case 4: Handling missing LGD values
    def mock_read_csv(*args, **kwargs):
        return pd.DataFrame({
            'default_quarter': ['2023-Q1', '2023-Q1', '2023-Q2', '2023-Q2'],
            'lgd': [0.1, None, 0.3, 0.4]
        })

    monkeypatch.setattr(pd, 'read_csv', mock_read_csv)

    result = aggregate_lgd_by_cohort()
    assert isinstance(result, pd.DataFrame)
    assert 'default_quarter' in result.columns
    assert 'mean_lgd' in result.columns
    assert len(result) == 2

    # Check values for 2023-Q1
    q1_data = result[result['default_quarter'] == '2023-Q1']['mean_lgd'].values[0]
    assert q1_data == 0.1

    # Check values for 2023-Q2
    q2_data = result[result['default_quarter'] == '2023-Q2']['mean_lgd'].values[0]
    assert q2_data == 0.35

def test_aggregate_lgd_by_cohort_different_quarters(monkeypatch):
    # Test case 5: Handling different default quarters (more than 2)
    def mock_read_csv(*args, **kwargs):
        return pd.DataFrame({
            'default_quarter': ['2022-Q4', '2023-Q1', '2023-Q2', '2023-Q3'],
            'lgd': [0.1, 0.2, 0.3, 0.4]
        })

    monkeypatch.setattr(pd, 'read_csv', mock_read_csv)

    result = aggregate_lgd_by_cohort()
    assert len(result) == 4  # Four unique quarters

    quarter_data = result.set_index('default_quarter')['mean_lgd'].to_dict()
    assert quarter_data['2022-Q4'] == 0.1
    assert quarter_data['2023-Q1'] == 0.2
    assert quarter_data['2023-Q2'] == 0.3
    assert quarter_data['2023-Q3'] == 0.4
