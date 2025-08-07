import pytest
import pandas as pd
from definition_c518a61a96b248f88a595a07960d88fd import compute_realized_lgd

@pytest.fixture
def sample_dataframe():
    # Create a sample DataFrame for testing
    data = {
        'EAD': [1000, 2000, 3000, 4000, 5000],
        'recoveries': [100, 0, 1500, 500, 6000],
        'collection_costs': [50, 100, 50, 200, 500]
    }
    return pd.DataFrame(data)

def test_compute_realized_lgd_typical(sample_dataframe):
    # Test with typical values
    df = sample_dataframe.copy()
    result_df = compute_realized_lgd(df)
    expected_lgd = [(1000 - 100 - 50) / 1000, (2000 - 0 - 100) / 2000, (3000 - 1500 - 50) / 3000, (4000 - 500 - 200) / 4000, (5000-6000-500)/5000]
    expected_lgd = [max(0, lgd) for lgd in expected_lgd]

    assert 'LGD_realized' in result_df.columns
    for i in range(len(expected_lgd)):
        assert result_df['LGD_realized'][i] == expected_lgd[i]

def test_compute_realized_lgd_all_recoveries_greater_than_ead(sample_dataframe):
    # Test where recoveries + collection costs exceed EAD, LGD should be 0
    df = sample_dataframe.copy()
    df['recoveries'] = [1500, 2500, 3500, 4500, 5500]
    result_df = compute_realized_lgd(df)
    assert all(result_df['LGD_realized'] == 0)

def test_compute_realized_lgd_zero_ead(sample_dataframe):
    # Test with zero EAD, should return 0 to avoid division by zero
    df = sample_dataframe.copy()
    df['EAD'] = [0, 0, 0, 0, 0]
    result_df = compute_realized_lgd(df)
    assert all(result_df['LGD_realized'] == 0)

def test_compute_realized_lgd_negative_recoveries_costs(sample_dataframe):
    # Test with negative recoveries and collection costs.
    df = sample_dataframe.copy()
    df['recoveries'] = [-100, -200, -300, -400, -500]
    df['collection_costs'] = [-50, -100, -50, -200, -500]

    result_df = compute_realized_lgd(df)
    expected_lgd = [(1000 - (-100) - (-50)) / 1000, (2000 - (-200) - (-100)) / 2000, (3000 - (-300) - (-50)) / 3000, (4000 - (-400) - (-200)) / 4000, (5000-(-500)-(-500))/5000]
    expected_lgd = [max(0, lgd) for lgd in expected_lgd]
    for i in range(len(expected_lgd)):
        assert result_df['LGD_realized'][i] == expected_lgd[i]

def test_compute_realized_lgd_empty_dataframe():
    # Test with an empty DataFrame
    df = pd.DataFrame()
    result_df = compute_realized_lgd(df)
    assert 'LGD_realized' not in result_df.columns if not result_df.empty else True