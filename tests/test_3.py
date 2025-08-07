import pytest
import pandas as pd
from definition_debe8b62fddc49108cae0491e9d3ced8 import assemble_recovery_cashflows

@pytest.fixture
def sample_df():
    # Create a sample DataFrame for testing
    data = {'loan_id': [1, 2, 3],
            'recovery_amount': [100, 200, 0],
            'collection_costs': [10, 20, 0],
            'outstanding_principal': [1000, 2000, 500]}
    return pd.DataFrame(data)

def test_assemble_recovery_cashflows_empty_df():
    df = pd.DataFrame()
    result = assemble_recovery_cashflows(df)
    assert isinstance(result, pd.DataFrame)

def test_assemble_recovery_cashflows_valid_data(sample_df):
    result = assemble_recovery_cashflows(sample_df)
    assert isinstance(result, pd.DataFrame)

def test_assemble_recovery_cashflows_no_recoveries(sample_df):
    sample_df['recovery_amount'] = 0
    sample_df['collection_costs'] = 0
    result = assemble_recovery_cashflows(sample_df)
    assert isinstance(result, pd.DataFrame)

def test_assemble_recovery_cashflows_negative_recoveries(sample_df):
    sample_df['recovery_amount'] = [-100, -200, -50]
    result = assemble_recovery_cashflows(sample_df)
    assert isinstance(result, pd.DataFrame)

def test_assemble_recovery_cashflows_missing_columns():
    df = pd.DataFrame({'loan_id': [1, 2, 3]})
    with pytest.raises(KeyError):  # Expect KeyError because 'recovery_amount' is missing
        assemble_recovery_cashflows(df)