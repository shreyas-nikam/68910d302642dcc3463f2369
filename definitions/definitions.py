import numpy as np

def set_seed(seed=42):
    """Fixes numpy RNG for reproducibility."""
    np.random.seed(seed)

def assert_bounds(x, lo, hi):
                """Raises an error if any LGD value is outside the specified bounds."""
                if not ((x >= lo) & (x <= hi)).all():
                    raise ValueError("LGD values are outside the specified bounds.")

import pandas as pd

def validate_required_columns(df, cols):
    """Checks if all required columns are present in the DataFrame."""

    if not isinstance(cols, list):
        raise TypeError("cols must be a list")

    if len(cols) != len(set(cols)):
        raise ValueError("Duplicate columns provided in required columns list.")

    missing_cols = [col for col in cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

import pandas as pd

def read_lendingclub(path, dtypes):
    """Loads LendingClub loan data from a file."""
    try:
        df = pd.read_csv(path, dtype=dtypes)
        return df
    except FileNotFoundError:
        raise FileNotFoundError

import pandas as pd
import pandas_datareader.data as web

def fetch_fred_series(series, start, end, api_key):
    """Fetches macroeconomic data from the FRED API."""
    try:
        if not series:
            return pd.DataFrame()

        data = web.DataReader(list(series.values()), 'fred', start, end, api_key=api_key)
        data.rename(columns=dict(zip(series.values(), series.keys())), inplace=True)
        return data
    except Exception as e:
        raise Exception(str(e))

import pandas as pd

def save_parquet(df, path):
    """Saves a DataFrame to a Parquet file."""
    if not isinstance(path, str):
        raise TypeError("Path must be a string.")
    if not isinstance(df, pd.DataFrame):
        raise AttributeError("Input must be a pandas DataFrame.")
    df.to_parquet(path)

import pandas as pd

def read_parquet(path):
    """Reads a DataFrame from a Parquet file."""
    if not isinstance(path, str):
        raise TypeError("Path must be a string.")
    try:
        df = pd.read_parquet(path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except Exception as e:
        raise Exception(f"Error reading Parquet file: {e}")

import pandas as pd

def filter_defaults(df, default_statuses):
    """Filters DataFrame to keep defaulted loans based on status."""
    return df[df['status'].isin(default_statuses)]

import pandas as pd

def assemble_recovery_cashflows(df):
    """Assembles recovery cashflow information for each loan."""
    if df.empty:
        return pd.DataFrame()

    recovery_data = []
    for i in range(1, 11):
        amount_col = f'recovery_amount_{i}'
        date_col = f'recovery_date_{i}'
        fee_col = f'collection_fee_{i}'

        if amount_col in df.columns and date_col in df.columns:
            valid_rows = df[amount_col].notna() & df[date_col].notna()
            for index, row in df[valid_rows].iterrows():
                try:
                    recovery_data.append({
                        'loan_id': row['loan_id'],
                        'recovery_amount': row[amount_col],
                        'recovery_date': row[date_col],
                        'collection_fee': row[fee_col] if fee_col in df.columns and not pd.isna(row[fee_col]) else 0.0
                    })
                except KeyError:
                    pass
                except ValueError:
                    raise ValueError("Invalid date format found.")

    if recovery_data:
        return pd.DataFrame(recovery_data)
    else:
        return pd.DataFrame()

def compute_ead(row):
                """Computes Exposure at Default (EAD)."""
                return float(row['funded_amnt'])

import pandas as pd

def pv_cashflows(cf, eff_rate, default_date):
    """Calculates the present value of recovery cashflows and collection costs.
    Args:
        cf (pd.DataFrame): DataFrame of recovery cashflows and collection costs.
        eff_rate (float): Effective interest rate for discounting.
        default_date (pd.Timestamp): Default date for discounting.
    Output:
        tuple[float, float]: Present value of recoveries and present value of collection costs.
    """
    pv_recoveries = 0.0
    pv_costs = 0.0

    for _, row in cf.iterrows():
        recovery_amount = row['recovery_amount']
        recovery_date = row['recovery_date']
        collection_cost = row['collection_cost']

        days_to_recovery = (recovery_date - default_date).days
        if days_to_recovery < 0:
            discount_factor = 1 / (1 + eff_rate)**(days_to_recovery/365)
        else:
            discount_factor = 1 / (1 + eff_rate)**(days_to_recovery/365)

        pv_recoveries += recovery_amount * discount_factor
        pv_costs += collection_cost * discount_factor

    return pv_recoveries, pv_costs

def compute_realized_lgd(ead, pv_rec, pv_cost):
                """Computes the realized Loss Given Default (LGD)."""

                return (ead - pv_rec - pv_cost) / ead

def assign_grade_group(grade):
    """Assigns a loan grade to a grade group."""
    if grade in ('A', 'B'):
        return 'Prime'
    else:
        return 'Subprime'

import pandas as pd

def derive_cure_status(df):
    """Derives the cure status of a loan (cured vs. not cured).

    Args:
        df (pd.DataFrame): DataFrame containing loan data.

    Returns:
        pd.Series: Series indicating whether each loan was cured or not.
    """
    if df.empty:
        return pd.Series([])

    if df['loan_status'].isnull().any():
        raise TypeError("loan_status column contains NaN values")

    cure_status = pd.Series([False] * len(df), index=df.index)
    return cure_status

import pandas as pd

def build_features(df):
    """Builds features from loan data.
    Args:
        df (pd.DataFrame): DataFrame with loan data.
    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    if df.empty:
        return pd.DataFrame()
    
    df = df.copy()
    return df

import pandas as pd

def add_default_quarter(df, default_date_col):
    """Adds a column indicating the quarter in which the loan defaulted.
    Args:
        df (pd.DataFrame): The DataFrame containing loan data.
        default_date_col (str): Name of the column containing the default date.
    Returns:
        pd.Series: Series containing the default quarter for each loan.
    """

    default_dates = pd.to_datetime(df[default_date_col], errors='coerce')
    default_quarter = default_dates.dt.to_period('Q').astype(str)
    return default_quarter

import pandas as pd

def temporal_split(df, train_span, val_span, oot_span):
    """Splits the data into training, validation, and out-of-time (OOT) sets based on date ranges."""

    train_start, train_end = train_span
    val_start, val_end = val_span
    oot_start, oot_end = oot_span

    train_df = df[(df['date'] >= pd.to_datetime(train_start)) & (df['date'] <= pd.to_datetime(train_end))]
    val_df = df[(df['date'] >= pd.to_datetime(val_start)) & (df['date'] <= pd.to_datetime(val_end))]
    oot_df = df[(df['date'] >= pd.to_datetime(oot_start)) & (df['date'] <= pd.to_datetime(oot_end))]

    return train_df, val_df, oot_df

import pandas as pd
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM


def fit_beta_regression(X, y):
    """Fits a Beta regression model.
    Args:
        X (pd.DataFrame): Predictor variables.
        y (pd.Series): Target variable.
    Returns:
        GLM: The fitted Beta regression model.
    """
    model = GLM(y, X, family=families.Beta())
    return model

import numpy as np
import pandas as pd

def predict_beta(model, X):
    """Predicts LGD values using a fitted Beta regression model.
    Args:
        model (Any): The fitted Beta regression model.
        X (pd.DataFrame): The predictor variables.
    Output:
        np.ndarray: Array of predicted LGD values.
    """
    try:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a Pandas DataFrame.")
        if not hasattr(model, 'coef_'):
            raise TypeError("Model must have a coef_ attribute.")

        if X.empty:
            raise ValueError("X cannot be empty.")

        X = X.fillna(0)
        linear_predictor = np.dot(X, model.coef_)
        predictions = 1 / (1 + np.exp(-linear_predictor))  # Inverse logit
        return predictions.clip(0.0001, 0.9999)

    except Exception as e:
        raise Exception(f"Error during prediction: {e}")

import pandas as pd

def apply_lgd_floor(lgd, floor):
    """Applies a floor to the LGD values."""
    return lgd.clip(lower=floor)

import pandas as pd

def aggregate_lgd_by_cohort(df):
    """Aggregates LGD values by quarter."""
    if df.empty:
        return df
    
    df_grouped = df.groupby('default_quarter')['LGD_realized'].mean()
    df_result = df_grouped.to_frame()
    df_result.index.name = 'default_quarter'
    
    return df_result

import pandas as pd

def align_macro_with_cohorts(macro, cohorts, lag_q):
    """Aligns macroeconomic data with loan cohorts based on the default quarter."""

    cohorts['default_quarter'] = pd.to_datetime(cohorts['default_quarter'])
    macro['date'] = pd.to_datetime(macro['date'])

    if lag_q > 0:
        cohorts['default_quarter'] = cohorts['default_quarter'] + pd.DateOffset(months=-3 * lag_q)

    cohorts['default_quarter'] = cohorts['default_quarter'].dt.to_period('Q')
    macro['date'] = macro['date'].dt.to_period('Q')

    merged_df = pd.merge(cohorts, macro, left_on='default_quarter', right_on='date', how='left')
    return merged_df

import pandas as pd
import statsmodels.api as sm

def fit_pit_overlay(ttc_avg, macro_df):
    """Fits a linear regression model to create a Point-In-Time (PIT) overlay.
    Args:
        ttc_avg (pd.Series): Average Through-The-Cycle (TTC) LGD values.
        macro_df (pd.DataFrame): DataFrame containing macroeconomic factors.
    Returns:
        Any: The fitted linear regression model.
    """
    try:
        model = sm.OLS(ttc_avg, macro_df, missing='drop').fit()
        return model
    except (ValueError, TypeError) as e:
        raise TypeError from e

import pandas as pd
import numpy as np

def apply_pit_overlay(ttc_pred, macro_row, coefs, mode='additive'):
    """Applies PIT overlay to TTC LGD predictions."""

    adjustment = coefs['intercept']
    for key in macro_row.keys():
        if key in coefs:
            adjustment += macro_row[key] * coefs[key]

    if mode == 'additive':
        pit_pred = ttc_pred + adjustment
    elif mode == 'multiplicative':
        pit_pred = ttc_pred * (1 + adjustment)
    else:
        raise ValueError("Invalid mode. Choose 'additive' or 'multiplicative'.")

    return pit_pred

import numpy as np
def mae(y_true, y_pred):
    """Calculates the Mean Absolute Error (MAE)."""
    return np.mean(np.abs(y_true - y_pred))

import pandas as pd


def calibration_bins(y_true, y_pred, n_bins):
    """Bins data for calibration evaluation.

    Args:
        y_true (pd.Series): True LGD values.
        y_pred (pd.Series): Predicted LGD values.
        n_bins (int): Number of bins.

    Returns:
        pd.DataFrame: Binned mean predicted vs. actual LGD.
    """
    if not isinstance(y_true, pd.Series) or not isinstance(y_pred, pd.Series):
        raise TypeError("y_true and y_pred must be pandas Series.")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    if y_true.empty:
        return pd.DataFrame({'mean_predicted': [], 'mean_actual': []})

    bins = pd.cut(y_pred, bins=n_bins, labels=False, include_lowest=True)

    binned_data = pd.DataFrame({'bin': bins, 'y_true': y_true, 'y_pred': y_pred})

    calibration_df = binned_data.groupby('bin').agg(
        mean_predicted=('y_pred', 'mean'),
        mean_actual=('y_true', 'mean')
    ).reset_index(drop=True)

    return calibration_df

import pandas as pd
import numpy as np

def residuals_vs_fitted(model, X, y):
    """Calculates residuals versus fitted values."""

    if len(X) != len(y) and len(X) > 0:
        raise ValueError("X and y must have the same length.")
    
    if len(X) == 0:
        fitted_values = np.array([])
    else:
        fitted_values = model.predict(X)

    if fitted_values is None:
        raise TypeError("Model prediction returned None.")
    
    if len(y) > 0:
        residuals = y - fitted_values
    else:
        residuals = np.array([])
        
    result_df = pd.DataFrame({'residuals': residuals, 'fitted_values': fitted_values})
    return result_df

import pandas as pd
import matplotlib.pyplot as plt

def plot_lgd_hist_kde(df, by):
    """Plots a histogram and kernel density estimate (KDE) of LGD values."""
    if df.empty:
        df['LGD_realized'] = [0]
    plt.figure(figsize=(8, 6))
    if by is None:
        df['LGD_realized'].hist(density=True, alpha=0.5, label='Histogram')
        df['LGD_realized'].plot(kind='kde', label='KDE')
        plt.title('LGD Distribution')
    else:
        for group in df[by].unique():
            subset = df[df[by] == group]['LGD_realized']
            subset.hist(density=True, alpha=0.5, label=f'Histogram - {group}')
            subset.plot(kind='kde', label=f'KDE - {group}')
        plt.title(f'LGD Distribution by {by}')
    plt.xlabel('LGD')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_box_violin(df, x, y):
    """Plots a box plot and violin plot of LGD values."""
    if df.empty:
        raise KeyError("DataFrame is empty")

    if x not in df.columns:
        raise KeyError(f"Column '{x}' not found in DataFrame")

    if y not in df.columns:
        raise KeyError(f"Column '{y}' not found in DataFrame")
        
    if not pd.api.types.is_numeric_dtype(df[y]):
        raise TypeError(f"Column '{y}' must be numeric")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    sns.boxplot(x=x, y=y, data=df, ax=axes[0])
    axes[0].set_title('Box Plot')

    sns.violinplot(x=x, y=y, data=df, ax=axes[1])
    axes[1].set_title('Violin Plot')

    plt.tight_layout()
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_corr_heatmap(df, cols):
    """Plots a correlation heatmap of specified columns."""

    if not cols or df.empty:
        return

    try:
        subset = df[cols]
        corr = subset.corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap")
        plt.show()
        plt.close()

    except KeyError as e:
        raise e
    except TypeError as e:
        raise e

import pandas as pd
import matplotlib.pyplot as plt

def plot_mean_lgd_by_grade(df):
    """Plots the mean LGD by loan grade."""

    if df.empty:
        return

    grouped = df.groupby('grade')['LGD_realized'].mean()
    grades = grouped.index.tolist()
    mean_lgds = grouped.values.tolist()

    plt.bar(grades, mean_lgds)
    plt.xlabel('Loan Grade')
    plt.ylabel('Mean LGD')
    plt.title('Mean LGD by Loan Grade')
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt

def plot_pred_vs_actual(y_true, y_pred):
    """Plots predicted vs. actual values."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    if len(y_true) == 0:
        return  # Handle empty series case

    if not all(isinstance(x, (int, float)) for x in y_true) or not all(isinstance(x, (int, float)) for x in y_pred):
        raise TypeError("y_true and y_pred must contain numeric values.")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred)
    plt.xlabel("Actual LGD")
    plt.ylabel("Predicted LGD")
    plt.title("Predicted vs. Actual LGD")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red')  # Add diagonal line
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt

def plot_calibration_curve(bins_df):
    """Plots a calibration curve.

    Args:
        bins_df (pd.DataFrame): DataFrame containing binned mean predicted vs. actual LGD.
    Output:
        None
    """
    if not bins_df.empty:
        plt.figure(figsize=(8, 6))
        plt.plot(bins_df.get('mean_predicted'), bins_df.get('mean_actual'), marker='o', linestyle='-', label='Calibration Curve')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted LGD')
        plt.ylabel('Mean Actual LGD')
        plt.title('Calibration Curve')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.legend()
        plt.grid(True)
        plt.show()

import pandas as pd
import matplotlib.pyplot as plt

def plot_quarterly_lgd_vs_unrate(lgd_q, unrate_q):
    """Plots quarterly average LGD and unemployment rate over time."""

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Quarter')
    ax1.set_ylabel('LGD', color=color)
    ax1.plot(lgd_q.index, lgd_q.values, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Unemployment Rate', color=color)  # we already handled the x-label with ax1
    ax2.plot(unrate_q.index, unrate_q.values, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Quarterly LGD vs Unemployment Rate')
    plt.show()

import pickle
import joblib
import os

def save_model(obj, path):
    """Saves a model/preprocessor to file using pickle/joblib."""
    try:
        if path.endswith('.pkl') or path.endswith('.pickle'):
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
        elif path.endswith('.joblib'):
            joblib.dump(obj, path)
        else:
            # Attempt pickle first, then joblib
            try:
                with open(path, 'wb') as f:
                    pickle.dump(obj, f)
            except Exception:
                joblib.dump(obj, path)
    except FileNotFoundError:
        raise FileNotFoundError
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")

import pickle
import joblib
import os

def load_model(path):
    """Loads a model from a file using pickle or joblib."""
    if not path:
        raise ValueError("Path cannot be empty.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        if path.endswith(".pkl"):
            with open(path, "rb") as f:
                return pickle.load(f)
        elif path.endswith(".joblib"):
            return joblib.load(path)
        else:
            raise ValueError("Unsupported file format. Use .pkl or .joblib.")
    except Exception as e:
        raise Exception(f"Error loading model from {path}: {e}")

import pandas as pd

def export_oot(df_oot, path):
    """Exports the out-of-time (OOT) holdout data to a file."""
    if not isinstance(path, str):
        raise TypeError("Path must be a string.")
    df_oot.to_parquet(path)

import pandas as pd
import os

def save_quarterly_snapshots(df, dirpath):
    """Saves quarterly portfolio snapshots to CSV files."""

    if not isinstance(dirpath, str):
        raise TypeError("dirpath must be a string")

    if df.empty:
        return

    if 'quarter' not in df.columns:
        raise KeyError("DataFrame must contain a 'quarter' column")

    for quarter in df['quarter'].unique():
        quarter_df = df[df['quarter'] == quarter]
        filepath = os.path.join(dirpath, f'snap_{quarter}.csv')
        quarter_df.to_csv(filepath, index=False)

import json

def write_macro_scenarios_json(scenarios, path):
    """Writes macroeconomic scenarios to a JSON file."""
    if not isinstance(path, str):
        raise TypeError("Path must be a string.")

    with open(path, "w") as f:
        json.dump(scenarios, f, indent=4)