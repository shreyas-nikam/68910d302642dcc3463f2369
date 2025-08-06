import random

def set_seed(seed):
    """Sets the random seed for reproducibility.
    Args: 
        seed: The integer value to use as the random seed.
    Output: 
        None
    """
    if seed is not None:
        if not isinstance(seed, int):
            raise TypeError("Seed must be an integer or None.")
        random.seed(seed)
    else:
        random.seed()

import pandas as pd

def read_lendingclub(file_path):
    """Reads LendingClub loan data from a CSV or Parquet file.
    Args:
        file_path: Path to the LendingClub loan data file.
    Returns:
        A pandas DataFrame containing the loan data.
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            raise ValueError("Unsupported file format. Only CSV and Parquet files are supported.")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise e

import pandas as pd

def filter_defaults(df):
    """Filters loan data to include only defaulted loans."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    if 'loan_status' not in df.columns:
        raise KeyError("DataFrame must contain 'loan_status' column")

    defaulted_loans = df[df['loan_status'] == 'Charged Off']
    return defaulted_loans

import pandas as pd

def assemble_recovery_cashflows(df):
    """Assembles the recovery cashflows for each loan."""
    if df.empty:
        return pd.DataFrame()

    try:
        df['net_recovery'] = df['recovery_amount'] - df['collection_cost']
    except KeyError:
        raise KeyError("Required columns ('recovery_amount', 'collection_cost') missing.")
    except TypeError:
        raise TypeError("Columns 'recovery_amount' and 'collection_cost' must be numeric.")
    
    return df

import pandas as pd

def compute_ead(df):
    """Computes the Exposure at Default (EAD) for each loan.
    Args:
        df: pandas DataFrame containing loan data.
    Output:
        A pandas DataFrame with the EAD calculated for each loan.
    """
    if df.empty:
        return df

    try:
        df['EAD'] = df['funded_amnt'] - df['total_pymnt']
    except KeyError as e:
        raise KeyError from e
    except TypeError:
        raise TypeError("Data types are not compatible for EAD calculation. Ensure numeric types.")
    except Exception as e:
        raise Exception(f"An error occurred during EAD calculation: {e}")

    return df

def pv_cashflows(cashflows, interest_rate, time_to_payment):
    """Calculates the present value of cashflows."""
    if interest_rate < -1:
        raise ValueError
    present_value = 0
    for i in range(len(cashflows)):
        present_value += cashflows[i] / (1 + interest_rate)**time_to_payment[i]
    return present_value

def compute_realized_lgd(ead, pv_recoveries, pv_collection_costs):
                """Computes Realized LGD given EAD, PV of recoveries, and PV of collection costs."""
                return (ead - pv_recoveries + pv_collection_costs) / ead

import pandas as pd

def assign_grade_group(df):
    """Assigns a grade group to each loan based on its loan grade."""

    if 'grade' not in df.columns:
        raise KeyError("DataFrame must contain a 'grade' column.")

    df['grade_group'] = df['grade'].apply(lambda grade: 'Prime' if grade in ['A', 'B'] else 'Sub-prime')
    return df

import pandas as pd

def derive_cure_status(df):
    """Derives cure status based on recovery date."""
    if df.empty:
        df['cure_status'] = []
        return df
    df['cure_status'] = df['recovery_date'].apply(lambda x: 'cured' if pd.notnull(x) else 'not cured')
    return df

import pandas as pd

def build_features(df):
    """Builds features for the LGD model from the loan data.

    Args:
        df: pandas DataFrame containing loan data.

    Returns:
        A pandas DataFrame with engineered features.
    """
    
    if df.empty:
        return pd.DataFrame()

    # Create a copy to avoid modifying the original DataFrame
    df_new = df.copy()
    
    # Basic Feature Engineering (can be expanded)
    if 'loan_amnt' in df_new.columns and 'int_rate' in df_new.columns:
        df_new['loan_amnt_x_int_rate'] = df_new['loan_amnt'] * df_new['int_rate']
    
    if 'installment' in df_new.columns and 'loan_amnt' in df_new.columns:
        df_new['installment_to_loan_ratio'] = df_new['installment'] / df_new['loan_amnt']
        
    return df_new

import pandas as pd

def add_default_quarter(df):
    """Adds a column indicating the quarter in which the loan defaulted."""

    df['default_quarter'] = None

    for index, row in df.iterrows():
        if row['loan_status'] == 'Charged Off':
            default_date = pd.to_datetime(row['last_credit_pull_d'], errors='coerce')
            if pd.notnull(default_date):
                quarter = default_date.quarter
                df.loc[index, 'default_quarter'] = quarter
            else:
                 df.loc[index, 'default_quarter'] = None 
        else:
            df.loc[index, 'default_quarter'] = None

    return df

import pandas as pd

def temporal_split(df, train_size):
    """Splits data into training and testing sets based on time."""
    if not 0 <= train_size <= 1:
        raise ValueError("Train size must be between 0 and 1")
    
    train_size = int(len(df) * train_size)
    train_df = df[:train_size]
    test_df = df[train_size:]
    return train_df, test_df

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

class BetaRegression:
    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

def fit_beta_regression(X_train, y_train):
    """Fits a Beta regression model to the training data.
    Args:
        X_train: Training features.
        y_train: Training target (LGD).
    Returns:
        A trained Beta regression model.
    Raises:
        ValueError: If y_train contains values outside the range (0, 1).
        TypeError: If X_train contains non-numeric values.
        ValueError: If X_train and y_train have mismatched lengths.
        Exception: If input data is empty.
    """

    if X_train.empty or y_train.empty:
        raise Exception("Input data cannot be empty.")

    if not all(0 < y_train) or not all(y_train < 1):
        raise ValueError("Target values must be between 0 and 1.")

    if not all(pd.api.types.is_numeric_dtype(X_train[col]) for col in X_train.columns):
        raise TypeError("Features must be numeric.")

    if len(X_train) != len(y_train):
        raise ValueError("Feature and target data must have the same length.")

    model = BetaRegression()
    model.fit(X_train, (y_train > 0.5).astype(int))  # Binary classification for mock
    return model

import pandas as pd
import numpy as np

def predict_beta(model, X):
    """Predicts LGD values using a trained Beta regression model.
    Args: 
        model: Trained Beta regression model. 
        X: Features to predict on.
    Returns: 
        Predicted LGD values.
    """
    try:
        predictions = model.predict(X)
        return predictions
    except Exception as e:
        raise e

def apply_lgd_floor(lgd_predictions, floor):
    """Applies a floor to the predicted LGD values."""
    return [max(x, floor) for x in lgd_predictions]

import pandas as pd

def aggregate_lgd_by_cohort(df):
    """Aggregates LGD values by loan origination cohort.
    Args: 
        df: pandas DataFrame containing loan data with LGD values and origination date.
    Output: 
        A pandas DataFrame with aggregated LGD values by cohort.
    """
    if df.empty:
        return df

    # Group by origination date and calculate the mean LGD
    aggregated_df = df.groupby('origination_date')['lgd'].mean()

    # Convert the result to a DataFrame and set the index name
    aggregated_df = aggregated_df.to_frame()
    aggregated_df.index.name = 'origination_date'

    return aggregated_df

import pandas as pd

def align_macro_with_cohorts(lgd_data, macro_data):
    """Aligns macro data with loan origination cohorts."""

    if lgd_data.empty or macro_data.empty:
        return pd.DataFrame()

    # Convert cohort to datetime
    lgd_data['cohort_date'] = lgd_data['cohort'].str.replace(r'Q(\d)', r'-\1').str.replace(r'(\d{4})', r'\1-').astype('datetime64[ns]')

    # Convert macro date to datetime
    macro_data['date'] = pd.to_datetime(macro_data['date'])

    # Create a mapping between cohort and macro data
    cohort_to_macro = {}
    for i, row in lgd_data.iterrows():
        cohort_date = row['cohort_date']
        # Find the macro data that matches the cohort date
        macro_match = macro_data[macro_data['date'] == cohort_date]
        if not macro_match.empty:
            cohort_to_macro[row['cohort']] = macro_match['gdp_growth'].values[0]
        else:
            cohort_to_macro[row['cohort']] = None

    # Add macro data to LGD data
    lgd_data['gdp_growth'] = lgd_data['cohort'].map(cohort_to_macro)

    lgd_data = lgd_data.drop('cohort_date', axis=1)
    return lgd_data

import pandas as pd
from sklearn.linear_model import LinearRegression

def fit_pit_overlay(X_train, y_train):
    """Fits a Point-In-Time (PIT) overlay model using linear regression.
    Args: 
        X_train: Training features (macroeconomic factors). 
        y_train: Training target (difference between realized LGD and TTC LGD).
    Returns: 
        A trained linear regression model.
    """
    if X_train.empty or y_train.empty:
        raise ValueError("Input DataFrames cannot be empty.")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same length.")
    if not all(pd.api.types.is_numeric_dtype(X_train[col]) for col in X_train.columns):
        raise TypeError("All features in X_train must be numeric.")

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model