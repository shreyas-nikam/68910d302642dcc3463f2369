import pytest
import pandas as pd
from definition_78108f9df9994cfa9a1c3114e5a75fe8 import segment_portfolio

@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'grade_group': ['A', 'B', 'A', 'C', 'B'],
        'cure_status': [True, False, True, False, True],
        'loan_amount': [1000, 2000, 1500, 2500, 1800]
    })
    return data

def test_segment_portfolio_empty_criteria(sample_data):
    segments = segment_portfolio(sample_data.copy(), {})
    assert len(segments) == 1
    assert 'All' in segments
    pd.testing.assert_frame_equal(segments['All'], sample_data)

def test_segment_portfolio_single_criteria(sample_data):
    segmentation_criteria = {'grade_group': ['A', 'B']}
    segments = segment_portfolio(sample_data.copy(), segmentation_criteria)
    assert len(segments) == 1
    assert 'grade_group_A_B' in segments
    expected_df = sample_data[sample_data['grade_group'].isin(['A', 'B'])]
    pd.testing.assert_frame_equal(segments['grade_group_A_B'], expected_df)

def test_segment_portfolio_multiple_criteria(sample_data):
    segmentation_criteria = {'grade_group': ['A', 'B'], 'cure_status': [True]}
    segments = segment_portfolio(sample_data.copy(), segmentation_criteria)
    assert len(segments) == 1
    assert 'grade_group_A_B_cure_status_True' in segments
    expected_df = sample_data[sample_data['grade_group'].isin(['A', 'B']) & sample_data['cure_status'].isin([True])]
    pd.testing.assert_frame_equal(segments['grade_group_A_B_cure_status_True'], expected_df)

def test_segment_portfolio_no_matching_data(sample_data):
    segmentation_criteria = {'grade_group': ['D', 'E']}
    segments = segment_portfolio(sample_data.copy(), segmentation_criteria)
    assert len(segments) == 1
    assert 'grade_group_D_E' in segments
    assert segments['grade_group_D_E'].empty

def test_segment_portfolio_invalid_data_type(sample_data):
    segmentation_criteria = {'loan_amount': ['invalid']}
    with pytest.raises(TypeError):
        segment_portfolio(sample_data.copy(), segmentation_criteria)
