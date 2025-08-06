import pytest
import pandas as pd
from definition_b329228b8b524ca6aa40f78a6e57ce0f import assemble_recovery_cashflows


def test_assemble_recovery_cashflows_empty_df():
    df = pd.DataFrame()
    result = assemble_recovery_cashflows(df)
    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_assemble_recovery_cashflows_no_recovery_columns(mocker):
    df = pd.DataFrame({'loan_id': [1, 2], 'default_date': ['2023-01-01', '2023-02-01']})
    mocker.patch('pandas.DataFrame', return_value=df) # Ensure a DataFrame is always returned, even when processing is empty
    result = assemble_recovery_cashflows(df)
    assert isinstance(result, pd.DataFrame)


def test_assemble_recovery_cashflows_valid_data(mocker):
    df = pd.DataFrame({
        'loan_id': [1, 2],
        'default_date': ['2023-01-01', '2023-02-01'],
        'recovery_amount_1': [100, 200],
        'recovery_date_1': ['2023-03-01', '2023-04-01'],
        'collection_fee_1': [10, 20]
    })
    mocker.patch('pandas.DataFrame', return_value=df)

    result = assemble_recovery_cashflows(df)
    assert isinstance(result, pd.DataFrame)


def test_assemble_recovery_cashflows_mixed_recovery_columns(mocker):
     df = pd.DataFrame({
        'loan_id': [1, 2],
        'default_date': ['2023-01-01', '2023-02-01'],
        'recovery_amount_1': [100, 200],
        'collection_fee_1': [10, 20]
    })
     mocker.patch('pandas.DataFrame', return_value=df)
     result = assemble_recovery_cashflows(df)
     assert isinstance(result, pd.DataFrame)



def test_assemble_recovery_cashflows_invalid_date_format(mocker):
    df = pd.DataFrame({
        'loan_id': [1],
        'default_date': ['invalid-date'],
        'recovery_amount_1': [100],
        'recovery_date_1': ['2023-03-01'],
        'collection_fee_1': [10]
    })
    mocker.patch('pandas.DataFrame', return_value=df)

    with pytest.raises(Exception):
        assemble_recovery_cashflows(df)
