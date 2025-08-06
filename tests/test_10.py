import pytest
import pandas as pd
from definition_423b13ad690b427a8b9df997908facfd import pv_cashflows

@pytest.fixture
def mock_dataframe():
    return pd.DataFrame({
        'recovery_amount': [100, 200, 300],
        'recovery_date': ['2024-01-15', '2024-02-15', '2024-03-15'],
        'collection_cost': [10, 20, 30]
    })

def test_pv_cashflows_basic(mock_dataframe):
    eff_rate = 0.05
    default_date = pd.Timestamp('2024-01-01')
    mock_dataframe['recovery_date'] = pd.to_datetime(mock_dataframe['recovery_date'])

    pv_recoveries, pv_costs = pv_cashflows(mock_dataframe, eff_rate, default_date)

    assert isinstance(pv_recoveries, float)
    assert isinstance(pv_costs, float)
    assert pv_recoveries > 0
    assert pv_costs > 0

def test_pv_cashflows_zero_rate(mock_dataframe):
    eff_rate = 0.0
    default_date = pd.Timestamp('2024-01-01')
    mock_dataframe['recovery_date'] = pd.to_datetime(mock_dataframe['recovery_date'])
    pv_recoveries, pv_costs = pv_cashflows(mock_dataframe, eff_rate, default_date)

    assert isinstance(pv_recoveries, float)
    assert isinstance(pv_costs, float)

def test_pv_cashflows_empty_dataframe():
    cf = pd.DataFrame({'recovery_amount': [], 'recovery_date': [], 'collection_cost': []})
    eff_rate = 0.05
    default_date = pd.Timestamp('2024-01-01')
    pv_recoveries, pv_costs = pv_cashflows(cf, eff_rate, default_date)
    assert pv_recoveries == 0.0
    assert pv_costs == 0.0

def test_pv_cashflows_different_date_format(mock_dataframe):
    eff_rate = 0.05
    default_date = pd.Timestamp('2023-12-01')
    mock_dataframe['recovery_date'] = pd.to_datetime(mock_dataframe['recovery_date'])

    pv_recoveries, pv_costs = pv_cashflows(mock_dataframe, eff_rate, default_date)

    assert isinstance(pv_recoveries, float)
    assert isinstance(pv_costs, float)
    assert pv_recoveries > 0
    assert pv_costs > 0

def test_pv_cashflows_negative_values():
    cf = pd.DataFrame({'recovery_amount': [-100, -200], 'recovery_date': ['2024-01-15', '2024-02-15'], 'collection_cost': [-10, -20]})
    cf['recovery_date'] = pd.to_datetime(cf['recovery_date'])
    eff_rate = 0.05
    default_date = pd.Timestamp('2024-01-01')

    pv_recoveries, pv_costs = pv_cashflows(cf, eff_rate, default_date)

    assert isinstance(pv_recoveries, float)
    assert isinstance(pv_costs, float)
    assert pv_recoveries < 0
    assert pv_costs < 0
