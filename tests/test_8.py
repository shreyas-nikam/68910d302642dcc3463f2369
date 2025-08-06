import pytest
import pandas as pd
from definition_308c0071d89a4009b052063ffd43573f import derive_cure_status

@pytest.fixture
def sample_dataframe():
    data = {'loan_id': [1, 2, 3, 4, 5],
            'recovery_date': ['2023-01-15', None, '2023-03-20', '2023-04-10', None]}
    return pd.DataFrame(data)

def test_derive_cure_status_no_recoveries(sample_dataframe):
    df = sample_dataframe.copy()
    df_result = derive_cure_status(df)
    assert 'cure_status' in df_result.columns
    assert df_result['cure_status'].tolist() == ['cured', 'not cured', 'cured', 'cured', 'not cured']

def test_derive_cure_status_all_recoveries(sample_dataframe):
    df = sample_dataframe.copy()
    df['recovery_date'] = ['2023-01-15'] * len(df)
    df_result = derive_cure_status(df)
    assert 'cure_status' in df_result.columns
    assert all(df_result['cure_status'] == 'cured')

def test_derive_cure_status_empty_dataframe():
    df = pd.DataFrame()
    df_result = derive_cure_status(df)
    assert 'cure_status' in df_result.columns
    assert len(df_result) == 0

def test_derive_cure_status_existing_column(sample_dataframe):
     df = sample_dataframe.copy()
     df['cure_status'] = ['initial'] * len(df)
     df_result = derive_cure_status(df)
     assert 'cure_status' in df_result.columns
     assert df_result['cure_status'].tolist() == ['cured', 'not cured', 'cured', 'cured', 'not cured']

def test_derive_cure_status_mixed_date_formats(sample_dataframe):
    df = sample_dataframe.copy()
    df['recovery_date'] = ['1/15/2023', None, '03-20-2023', '20230410', None]
    df_result = derive_cure_status(df)
    assert 'cure_status' in df_result.columns
    assert df_result['cure_status'].tolist() == ['cured', 'not cured', 'cured', 'cured', 'not cured']
