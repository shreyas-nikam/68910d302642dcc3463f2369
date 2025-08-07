import pytest
import pandas as pd
from definition_b730811bc0ed4cf89d3bb75ac01ad585 import build_features

@pytest.fixture
def sample_dataframe():
    data = {'loan_amnt': [10000, 20000, 15000],
            'int_rate': [0.10, 0.12, 0.08],
            'annual_inc': [60000, 80000, 70000]}
    return pd.DataFrame(data)

def test_build_features_empty_features(sample_dataframe):
    """Test when an empty list of features is provided."""
    df = build_features(sample_dataframe, [])
    assert df.equals(sample_dataframe), "Should return the original DataFrame unchanged."

def test_build_features_invalid_feature(sample_dataframe):
    """Test when an invalid feature is requested."""
    with pytest.raises(KeyError):
        build_features(sample_dataframe, ['invalid_feature'])

def test_build_features_single_feature(sample_dataframe):
    """Test building a single simple feature (e.g., creating a derived column)."""
    def feature_func(df):
        df['loan_size_income_ratio'] = df['loan_amnt'] / df['annual_inc']
        return df
    
    # Mock the feature implementation by directly modifying the DataFrame in build_features
    def mock_build_features(df, features):
        if 'loan_size_income_ratio' in features:
            return feature_func(df)
        else:
            return df

    # Replace build_features temporarily for the purpose of this test
    import definition_b730811bc0ed4cf89d3bb75ac01ad585
    original_build_features = definition_b730811bc0ed4cf89d3bb75ac01ad585.build_features
    definition_b730811bc0ed4cf89d3bb75ac01ad585.build_features = mock_build_features

    try:
        df = build_features(sample_dataframe, ['loan_size_income_ratio'])
        assert 'loan_size_income_ratio' in df.columns, "Feature should be added."
        assert df['loan_size_income_ratio'].iloc[0] == 10000 / 60000, "Ratio should be calculated correctly."
    finally:
        # Restore the original build_features function
        definition_b730811bc0ed4cf89d3bb75ac01ad585.build_features = original_build_features

def test_build_features_multiple_features(sample_dataframe):
    """Test building multiple features."""
    def feature_func_1(df):
        df['loan_size_income_ratio'] = df['loan_amnt'] / df['annual_inc']
        return df

    def feature_func_2(df):
        df['int_rate_squared'] = df['int_rate']**2
        return df
    
    # Mock the feature implementation by directly modifying the DataFrame in build_features
    def mock_build_features(df, features):
        if 'loan_size_income_ratio' in features:
            df = feature_func_1(df)
        if 'int_rate_squared' in features:
            df = feature_func_2(df)
        return df

    # Replace build_features temporarily for the purpose of this test
    import definition_b730811bc0ed4cf89d3bb75ac01ad585
    original_build_features = definition_b730811bc0ed4cf89d3bb75ac01ad585.build_features
    definition_b730811bc0ed4cf89d3bb75ac01ad585.build_features = mock_build_features

    try:
        features_to_build = ['loan_size_income_ratio', 'int_rate_squared']
        df = build_features(sample_dataframe, features_to_build)
        assert 'loan_size_income_ratio' in df.columns, "Feature 1 should be added."
        assert 'int_rate_squared' in df.columns, "Feature 2 should be added."
        assert df['loan_size_income_ratio'].iloc[0] == 10000 / 60000, "Ratio should be calculated correctly."
        assert df['int_rate_squared'].iloc[0] == 0.10**2, "Squared rate should be calculated correctly."
    finally:
        # Restore the original build_features function
        definition_b730811bc0ed4cf89d3bb75ac01ad585.build_features = original_build_features

def test_build_features_dataframe_immutability(sample_dataframe):
    """Test that the original DataFrame is not modified if no features are built."""
    original_df = sample_dataframe.copy()
    build_features(sample_dataframe, [])
    pd.testing.assert_frame_equal(sample_dataframe, original_df, "Original DataFrame should not be modified.")
