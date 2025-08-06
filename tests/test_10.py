import pytest
import pandas as pd
from definition_abffc4c7caca4dbaab980b6e1acb981d import add_default_quarter

def test_add_default_quarter_empty_df():
    df = pd.DataFrame()
    result_df = add_default_quarter(df)
    assert 'default_quarter' in result_df.columns if not result_df.empty else True

def test_add_default_quarter_no_default_date():
    df = pd.DataFrame({'issue_d': ['2023-01-01'], 'loan_status': ['Fully Paid']})
    result_df = add_default_quarter(df)
    assert 'default_quarter' in result_df.columns if not result_df.empty else True

def test_add_default_quarter_valid_default_date():
    df = pd.DataFrame({'issue_d': ['2023-01-01'], 'loan_status': ['Charged Off'], 'last_credit_pull_d': ['2024-03-15']})
    result_df = add_default_quarter(df)
    assert 'default_quarter' in result_df.columns
    assert not result_df['default_quarter'].isnull().any()

def test_add_default_quarter_mixed_loan_status():
    df = pd.DataFrame({
        'issue_d': ['2023-01-01', '2023-04-01', '2023-07-01'],
        'loan_status': ['Charged Off', 'Fully Paid', 'Charged Off'],
        'last_credit_pull_d': ['2024-03-15', '2023-06-01', '2024-01-10']
    })
    result_df = add_default_quarter(df)
    assert 'default_quarter' in result_df.columns

def test_add_default_quarter_incorrect_date_format():
     df = pd.DataFrame({'issue_d': ['2023/01/01'], 'loan_status': ['Charged Off'], 'last_credit_pull_d': ['2024/03/15']})
     result_df = add_default_quarter(df)
     assert 'default_quarter' in result_df.columns

