import datetime

def calculate_realized_lgd(ead, recoveries, collection_costs, interest_rate, recovery_dates, cost_dates, default_date):
    """Calculates the realized Loss Given Default (LGD) for a given loan."""

    pv_recoveries = sum([r / (1 + interest_rate)**((d - default_date).days / 365.25) for r, d in zip(recoveries, recovery_dates)])
    pv_costs = sum([c / (1 + interest_rate)**((d - default_date).days / 365.25) for c, d in zip(collection_costs, cost_dates)])

    if ead:
        lgd = (ead - pv_recoveries - pv_costs) / ead
    else:
        lgd = 0.0

    return max(0.0, lgd)

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

def train_beta_regression_model(data, features, target):
    """Trains a Beta regression model.
    Args:
        data: Pandas DataFrame.
        features: List of feature names.
        target: Target variable name.
    Returns:
        Trained Beta regression model.
    Raises:
        ValueError: If features are empty or target is out of range.
        KeyError: If target column is missing.
        TypeError: If data is not a DataFrame.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a Pandas DataFrame.")

    if not features:
        raise ValueError("Features cannot be empty.")

    if target not in data.columns:
        raise KeyError("Target column missing from data.")
    
    if not all(0 < data[target] < 1):
        raise ValueError("Target values must be between 0 and 1.")

    formula = f"{target} ~ " + " + ".join(features)
    model = smf.beta(formula, data=data).fit()
    return model

import pandas as pd
from sklearn.linear_model import LinearRegression

def train_pit_overlay_model(ttc_lgd, macroeconomic_data, macroeconomic_features):
    """Trains a linear regression model for PIT overlay."""
    if not isinstance(ttc_lgd, pd.Series) or not isinstance(macroeconomic_data, pd.DataFrame) or not isinstance(macroeconomic_features, list):
        raise TypeError("Invalid input types.")

    if macroeconomic_data.empty and macroeconomic_features:
        raise ValueError("Macroeconomic data is empty but features are provided.")
    
    for feature in macroeconomic_features:
        if feature not in macroeconomic_data.columns:
            raise ValueError(f"Macroeconomic feature '{feature}' not found in data.")

    if macroeconomic_features:
        X = macroeconomic_data[macroeconomic_features]
    else:
        X = pd.DataFrame(index=ttc_lgd.index)  # Create an empty DataFrame with the correct index

    y = ttc_lgd

    if X.empty and y.empty:
        return LinearRegression()
    
    model = LinearRegression()
    model.fit(X, y)
    return model

import numpy as np
from sklearn.metrics import mean_absolute_error

def calculate_model_evaluation_metrics(y_true, y_pred):
    """Calculates model evaluation metrics such as pseudo-R-squared and MAE.
    Args:
        y_true: Array-like of actual LGD values.
        y_pred: Array-like of predicted LGD values.
    Output:
        Dictionary of evaluation metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate pseudo-R-squared
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    pseudo_r_squared = 1 - (numerator / denominator) if denominator != 0 else 0

    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true, y_pred)

    return {'pseudo_r_squared': float(pseudo_r_squared), 'mae': float(mae)}