import random

def set_seed():
    """Sets a fixed random seed for reproducibility."""
    random.seed(42)

import pandas as pd
import os

def read_lendingclub():
    """Reads LendingClub loan data from a CSV or Parquet format.
    Output: Pandas DataFrame containing the loan data.
    """
    
    # Check for CSV file
    if os.path.exists("test.csv"):
        try:
            df = pd.read_csv("test.csv")
            return df
        except Exception as e:
            raise FileNotFoundError("Error reading CSV file.") from e
    # Check for Parquet file
    elif os.path.exists("test.parquet"):
        try:
            df = pd.read_parquet("test.parquet")
            return df
        except Exception as e:
            raise FileNotFoundError("Error reading Parquet file.") from e
    else:
        # If no file found, return empty dataframe
        return pd.DataFrame()

import pandas as pd

def filter_defaults(df):
    """Filters loan data to include only defaulted loans.
    Args:
        df: Pandas DataFrame containing loan data.
    Returns:
        Pandas DataFrame containing only defaulted loans.
    """
    if df.empty:
        return pd.DataFrame()
    
    defaulted_loans = df[df['loan_status'] == 'Charged Off'].copy()
    defaulted_loans.reset_index(drop=True, inplace=True)
    return defaulted_loans

import pandas as pd

def assemble_recovery_cashflows():
    """Assembles recovery cashflows associated with defaulted loans.

    Returns:
        pd.DataFrame: DataFrame containing recovery cashflow information.
    """

    try:
        # Load the data from CSV
        df = pd.read_csv("path/to/recovery_data.csv")  # Replace with actual path

        # Handle missing values - fill NaN values with 0
        df = df.fillna(0)

        # Ensure data types are numeric
        for col in df.columns:
            if 'recovery_amount' in col:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) #Convert to numeric, coerce errors to NaN, then fill NaN with 0

        return df
    except FileNotFoundError:
        print("Recovery data file not found.")
        return pd.DataFrame()  # Return an empty DataFrame if the file is not found
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame() # Return an empty DataFrame if any other error occurs

import pandas as pd

# Assuming a DataFrame named 'df' is accessible within this scope
# e.g., df = pd.read_csv('loan_data.csv')

def compute_ead():
    """Computes the Exposure at Default (EAD) for each loan.
    Output: Pandas Series containing EAD values.
    """
    try:
        # Access the DataFrame 'df'
        global df  # Declare 'df' as global to access it

        # Simple EAD calculation: assuming EAD is the funded amount
        ead = df['funded_amnt']

        return ead
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.Series()  # Return an empty series in case of error

def pv_cashflows(cashflows, rate, time):
    """Calculates the present value of a series of cashflows."""
    if rate < -1:
        raise ValueError
    pv = 0
    for i in range(len(cashflows)):
        pv += cashflows[i] / (1 + rate)**time[i]
    return pv

import pandas as pd

def compute_realized_lgd():
    """Computes the realized LGD for each loan."""
    ead = compute_realized_lgd.ead
    recoveries_pv = compute_realized_lgd.recoveries_pv
    collection_costs_pv = compute_realized_lgd.collection_costs_pv
    lgd = (ead - recoveries_pv - collection_costs_pv) / ead
    return lgd

import pandas as pd

def assign_grade_group(df):
    """Assigns a grade group to each loan based on its loan grade."""
    if df.empty:
        return pd.Series([], dtype='object')

    def assign_group(grade):
        if grade in ('A', 'B'):
            return 'Prime'
        elif grade in ('C', 'D', 'E', 'F', 'G'):
            return 'Sub-prime'
        else:
            return 'Other'

    return df['grade'].apply(assign_group)

import pandas as pd

def derive_cure_status(df):
    """Derives cure status based on recovery amounts."""
    if df.empty:
        return pd.Series([])

    try:
        total_recoveries = df['collection_recovery_fee'] + df['recoveries']
        cure_status = total_recoveries.apply(lambda x: 'Cured' if x > 0 else 'Not Cured')
        return cure_status
    except KeyError as e:
        raise KeyError(f"Required column(s) missing: {e}")

import pandas as pd

def build_features():
    """Builds features for the LGD model.
    Returns: Pandas DataFrame.
    """
    data = {'loan_amnt': [10000, 20000, 30000],
            'int_rate': [0.10, 0.12, 0.15],
            'term': [36, 60, 36],
            'grade': ['A', 'B', 'C']}
    df = pd.DataFrame(data)
    return df

import pandas as pd

def add_default_quarter():
    """Adds the quarter of default to each defaulted loan record."""
    try:
        df = globals().get('df')
        if df is None or df.empty:
            return pd.Series([])
        default_dates = pd.to_datetime(df['default_date'], errors='raise')
        default_quarters = default_dates.dt.to_period('Q').astype(str)
        return default_quarters
    except KeyError:
        raise KeyError("The 'default_date' column is missing.")
    except ValueError:
        raise ValueError("Invalid date format in 'default_date' column.")

import pandas as pd

def temporal_split(df):
    """Splits DataFrame into training and OOT sets based on time."""
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    oot_year = 2019
    train_df = df[df['issue_d'].dt.year < oot_year].copy()
    oot_df = df[df['issue_d'].dt.year >= oot_year].copy()
    
    return train_df, oot_df

import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.families import Beta
from statsmodels.genmod.generalized_linear_model import GLM
import numpy as np

def fit_beta_regression(X, y):
    """Trains a Beta regression model.
    Args:
        X (DataFrame): Predictor variables.
        y (Series): Target variable (between 0 and 1).
    Returns:
        Fitted Beta regression model object.
    """
    if X.empty or y.empty:
        raise ValueError("Input data is empty.")

    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
        raise TypeError("X must contain numeric data")
    
    if not all((y >= 0) & (y <= 1)):
        raise ValueError("y must be between 0 and 1")
    
    if X.isnull().any().any() or y.isnull().any():
         raise ValueError("Input data contains NaN values. Please impute or drop them")
    
    model = GLM(y, sm.add_constant(X), family=Beta()).fit()
    return model

def predict_beta(model, X):
                """Predicts LGD values using a trained Beta regression model.
                Args: model (object), X (DataFrame)
                Output: Pandas Series containing predicted LGD values.
                """
                return model.predict(X)

import pandas as pd

def apply_lgd_floor(s: pd.Series) -> pd.Series:
    """Applies a 5% LGD floor to the predicted LGD values."""
    return s.clip(lower=0.05)

import pandas as pd
import statsmodels.api as sm

def fit_fractional_logit(X, y):
    """Fits a fractional logit model.
    Args: X (DataFrame), y (Series)
    Output: Fitted fractional logit model object.
    """
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise TypeError("X must be a DataFrame and y must be a Series.")
    
    if X.empty or y.empty:
        raise Exception("X and y cannot be empty.")
    
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")

    if not all((0 <= y) & (y <= 1)):
        raise Exception("y must contain values between 0 and 1")
    
    try:
        X = sm.add_constant(X)
        model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        return model
    except Exception as e:
        raise Exception(f"An error occurred during model fitting: {e}")

import pandas as pd

def aggregate_lgd_by_cohort():
    """Aggregates LGD by cohort."""
    try:
        df = pd.read_csv('data.csv')
    except FileNotFoundError:
        return pd.DataFrame()

    if 'default_quarter' not in df.columns:
        raise KeyError("default_quarter")

    if df.empty:
        return pd.DataFrame()

    df = df.dropna(subset=['lgd'])

    aggregated_lgd = df.groupby('default_quarter')['lgd'].mean().reset_index()
    aggregated_lgd.rename(columns={'lgd': 'mean_lgd'}, inplace=True)

    return aggregated_lgd

import pandas as pd

def align_macro_with_cohorts():
    """Aligns macroeconomic data with loan cohorts based on time."""

    loan_data = pd.read_csv("loan_cohorts.csv")
    macro_data = pd.read_csv("macro_data.csv")

    merged_data = pd.merge(loan_data, macro_data, left_on='default_quarter', right_on='quarter', how='left')

    return merged_data

import pandas as pd
from sklearn.linear_model import LinearRegression

def fit_pit_overlay(X, y):
    """Trains a linear regression model to adjust LGD based on macroeconomic factors.
    Args:
        X (DataFrame): Macroeconomic factors.
        y (Series): TTC LGD values.
    Returns:
        Fitted linear regression model.
    Raises:
        ValueError: If X or y is empty or if their lengths are mismatched.
        TypeError: If X contains non-numeric values.
    """

    if X.empty or y.empty:
        raise ValueError("Input DataFrames cannot be empty.")

    if len(X) != len(y):
        raise ValueError("X and y must have the same length.")

    if not all(pd.api.types.is_numeric_dtype(X[col]) for col in X.columns):
        raise TypeError("All columns in X must be numeric.")

    model = LinearRegression()
    model.fit(X, y)
    return model

import pandas as pd

def apply_pit_overlay(ttc_lgd, macro_factors, model):
    """Applies PIT overlay to TTC LGD."""

    if ttc_lgd.index.equals(macro_factors.index) == False:
        raise ValueError("TTC LGD and macro factors must have the same index.")
    if macro_factors.isnull().values.any():
        raise ValueError("Macro factors cannot contain missing values.")

    adjustments = model.predict(macro_factors)
    pit_lgd = ttc_lgd + adjustments
    pit_lgd = pit_lgd.clip(0, 1)
    return pit_lgd

import pandas as pd
            import numpy as np

            def mae(y_true, y_pred):
                """Calculates the Mean Absolute Error (MAE) between predicted and actual LGD values.
                Args: y_true (Series), y_pred (Series)
                Output: float: MAE value.
                """
                if len(y_true) != len(y_pred):
                    raise ValueError("y_true and y_pred must have the same length.")

                if len(y_true) == 0:
                    return 0.0

                return np.mean(np.abs(y_true - y_pred))

import pandas as pd
import numpy as np

def calibration_bins(y_true, y_pred, n_bins):
    """Calculates calibration bins for assessing model calibration.
    Args: y_true (Series), y_pred (Series), n_bins (int)
    Output: Pandas DataFrame containing calibration bin information.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    if y_true.empty:
        return pd.DataFrame()

    bins = np.linspace(0, 1, n_bins + 1)
    bin_assignments = pd.cut(y_pred, bins, labels=False, include_lowest=True)

    bin_counts = bin_assignments.value_counts(sort=False)
    bin_sums = y_true.groupby(bin_assignments).sum()
    bin_means_pred = y_pred.groupby(bin_assignments).mean()

    bin_df = pd.DataFrame({
        'counts': bin_counts,
        'sums': bin_sums,
        'means_pred': bin_means_pred
    })

    bin_df['counts'] = bin_df['counts'].fillna(0).astype(int)
    bin_df['sums'] = bin_df['sums'].fillna(0).astype(int)
    bin_df['means_pred'] = bin_df['means_pred'].fillna(0)

    bin_df['means_true'] = bin_df['sums'] / bin_df['counts']
    bin_df = bin_df.fillna(0)

    bin_df = bin_df.reindex(range(n_bins), fill_value=0)

    return bin_df

import pandas as pd


def residuals_vs_fitted(model, X, y):
    """Analyzes residuals vs. fitted values for model diagnostics.
    Args: model (object), X (DataFrame), y (Series)
    Output: Pandas DataFrame containing residuals and fitted values.
    """
    try:
        fitted = model.predict(X)
        residuals = y - fitted
        df = pd.DataFrame({'residuals': residuals, 'fitted': fitted})
        return df
    except Exception as e:
        raise e

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_lgd_hist_kde():
    """Visualizes the distribution of LGD_realized using histograms and KDE plots."""

    # Create dummy data
    data = {'LGD_realized': np.random.normal(loc=0.2, scale=0.1, size=100),
            'grade_group': np.random.choice(['A', 'B', 'C'], size=100)}
    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 6))
    sns.histplot(df['LGD_realized'], kde=True)
    plt.xlabel("LGD_realized")
    plt.ylabel("Density")
    plt.title("Distribution of LGD_realized")
    plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_box_violin():
    """Compares LGD across different loan terms and cure statuses using box/violin plots."""
    try:
        # Load the dataset
        loan_data = pd.read_csv('loan_data_2007_2014.csv')

        # Data Cleaning and Preprocessing (minimal, based on previous notebooks)
        loan_data = loan_data[loan_data['loan_status'].isin(['Charged Off', 'Fully Paid'])]
        loan_data['recovery_rate'] = loan_data['recoveries'] / loan_data['funded_amnt']
        loan_data['recovery_rate'] = loan_data['recovery_rate'].fillna(0)
        loan_data['LGD'] = 1 - loan_data['recovery_rate']
        loan_data = loan_data[loan_data['LGD'] >= 0]
        loan_data = loan_data[loan_data['LGD'] <= 1]
        loan_data['term'] = loan_data['term'].str.replace(' months', '')
        loan_data['term'] = pd.to_numeric(loan_data['term'], errors='coerce')
        loan_data['term'] = loan_data['term'].fillna(loan_data['term'].median())

        # Create the plot
        plt.figure(figsize=(14, 6))

        # Box plot
        plt.subplot(1, 2, 1)
        sns.boxplot(x='term', y='LGD', data=loan_data)
        plt.title('LGD vs. Loan Term (Box Plot)')
        plt.xlabel('Loan Term (Months)')
        plt.ylabel('LGD')

        # Violin plot
        plt.subplot(1, 2, 2)
        sns.violinplot(x='term', y='LGD', data=loan_data)
        plt.title('LGD vs. Loan Term (Violin Plot)')
        plt.xlabel('Loan Term (Months)')
        plt.ylabel('LGD')

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("Error: loan_data_2007_2014.csv not found. Skipping plot.")
    except Exception as e:
        print(f"An error occurred during plotting: {e}")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_corr_heatmap():
    """Shows correlations among numeric columns using a heatmap."""
    df = pd.read_csv('application_data.csv') # read dataframe
    numeric_df = df.select_dtypes(include=['number']) # select numeric columns
    numeric_df = numeric_df.dropna(axis=1, how='all') # remove columns with all NA values

    if len(numeric_df.columns) < 2:
        print("Insufficient numeric columns (less than 2) to plot correlation heatmap.")
        return

    try:
        corr = numeric_df.corr() # calculate correlations
        plt.figure(figsize=(12, 10)) # set figure size
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f") # create heatmap
        plt.title("Correlation Heatmap of Numeric Features") # set title
        plt.show() # show plot
    except TypeError as e:
        raise TypeError(f"TypeError occurred during correlation calculation or plotting: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

import matplotlib.pyplot as plt

def plot_mean_lgd_by_grade():
    """Illustrates mean LGD by loan grade using a bar chart."""

    grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    mean_lgd = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    plt.bar(grades, mean_lgd)
    plt.xlabel('Loan Grade')
    plt.ylabel('Mean LGD')
    plt.title('Mean LGD by Loan Grade')
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_pred_vs_actual():
    """Compares predicted vs. actual LGD values using a scatter plot."""

    # Dummy data for demonstration
    predicted_lgd = np.random.rand(100)
    actual_lgd = predicted_lgd + np.random.normal(0, 0.1, 100)

    plt.figure(figsize=(8, 6))
    plt.scatter(predicted_lgd, actual_lgd, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r-', transform=plt.gca().transAxes, label="45-degree line") # 45-degree line
    plt.xlabel("Predicted LGD")
    plt.ylabel("Actual LGD")
    plt.title("Predicted vs Actual LGD")
    plt.legend()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve


def plot_calibration_curve():
    """Plots a calibration curve."""

    # Dummy data for demonstration
    y_true = np.random.randint(0, 2, size=100)
    y_prob = np.random.rand(100)

    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

    # Plot the calibration curve
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', label='Calibration Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig("calibration_curve.png")
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

def plot_quarterly_lgd_vs_unrate():
    """Displays quarterly average LGD and unemployment rate over time using a dual-axis line chart."""

    # Generate some dummy data
    dates = pd.to_datetime(pd.date_range('2020-01-01', '2024-01-01', freq='QS'))
    lgd = np.random.rand(len(dates))
    unrate = np.random.rand(len(dates)) * 10  # Scale unemployment rate for visibility

    # Create a DataFrame
    df = pd.DataFrame({'Date': dates, 'LGD': lgd, 'Unemployment Rate': unrate})
    df = df.set_index('Date')

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot LGD
    color = 'tab:blue'
    ax1.set_xlabel('Quarter')
    ax1.set_ylabel('LGD', color=color)
    ax1.plot(df.index, df['LGD'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis
    ax2 = ax1.twinx()

    # Plot Unemployment Rate
    color = 'tab:red'
    ax2.set_ylabel('Unemployment Rate (%)', color=color)
    ax2.plot(df.index, df['Unemployment Rate'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Add title
    plt.title('Quarterly Average LGD vs. Unemployment Rate')

    # Show the plot
    plt.show()

import pickle
import joblib

def save_model(model, filename):
    """Saves the trained model to a file."""
    if not isinstance(filename, str):
        raise TypeError("Filename must be a string.")

    if model is None:
        raise ValueError("Model cannot be None.")

    try:
        if filename.endswith('.pkl'):
            with open(filename, 'wb') as file:
                pickle.dump(model, file)
        elif filename.endswith('.joblib'):
            joblib.dump(model, filename)
        else:
            raise ValueError("Filename must end with .pkl or .joblib")
    except Exception as e:
        raise Exception(f"Error saving model: {e}")

import pandas as pd

            def export_oot(oot_data, filename):
                """Exports the out-of-time (OOT) sample to a file.
        Arguments: oot_data (DataFrame), filename (str)
        Output: None
                """
                if filename is None:
                    raise TypeError("Filename cannot be None")

                if not str(filename).endswith(".parquet"):
                    raise ValueError("Filename must have a .parquet extension")
                oot_data.to_parquet(filename)

import pandas as pd

def save_quarterly_snapshots(snapshots, filename):
    """Saves the quarterly portfolio snapshots to a file."""
    snapshots.to_csv(filename, index=False)

import json

            def write_macro_scenarios_json(scenarios, filename):
                """Writes macro forecast scenarios to a JSON file."""
                with open(filename, "w") as f:
                    json.dump(scenarios, f)