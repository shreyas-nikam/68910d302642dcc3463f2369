import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error
import joblib # For saving/loading models

# Placeholder for macroeconomic data (replace with actual data loading)
def load_macroeconomic_data():
    # Example: GDP growth rate, unemployment rate
    macro_data = pd.DataFrame({
        'quarter': pd.to_datetime(['2018-Q4', '2019-Q1', '2019-Q2', '2019-Q3', '2019-Q4']),
        'unemployment_rate': [3.7, 3.8, 3.6, 3.5, 3.6],
        'gdp_growth': [2.5, 2.8, 2.2, 2.0, 2.1]
    })
    macro_data['quarter'] = macro_data['quarter'].dt.to_period('Q') # Convert to quarter period
    macro_data['quarter_str'] = macro_data['quarter'].astype(str)  # String representation for merging
    return macro_data

def align_macro_with_cohorts(lgd_cohorts, macro_data):
    """
    Aligns macroeconomic data with LGD cohorts based on time.
    Arguments:
        lgd_cohorts: Pandas DataFrame with LGD cohorts and a 'default_quarter_str' column.
        macro_data: Pandas DataFrame containing macroeconomic data with a 'quarter_str' column.
    Output:
        Pandas DataFrame with LGD cohorts and aligned macroeconomic data.
    """
    # Merge LGD cohorts with macroeconomic data based on the 'default_quarter_str' and 'quarter_str' columns
    aligned_data = pd.merge(lgd_cohorts, macro_data, left_on='default_quarter_str', right_on='quarter_str', how='left')
    return aligned_data

def fit_pit_overlay(X_train, y_train):
    """
    Fits a Point-In-Time (PIT) overlay model to adjust the TTC LGD based on macroeconomic factors.
    Arguments:
        X_train: Training features (macroeconomic variables).
        y_train: Training target variable (difference between realized LGD and TTC LGD).
    Output:
        Trained PIT overlay model (statsmodels OLS object).
    """
    # Add a constant for the intercept
    X_train_sm = sm.add_constant(X_train)

    try:
        # Fit an ordinary least squares (OLS) regression model
        model = sm.OLS(y_train, X_train_sm).fit()
        st.success(