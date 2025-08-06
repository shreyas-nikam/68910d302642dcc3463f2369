import pandas as pd

def load_and_preprocess_data(file_path):
    """Loads, cleans, and preprocesses data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df = df.dropna()
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

def calculate_lgd(ead, recoveries):
                """Calculates LGD, clipping values between 0 and 1."""
                if ead == 0:
                    return 0.0
                lgd = 1.0 - (recoveries / ead)
                return max(0.0, min(lgd, 1.0))

import pandas as pd

def segment_portfolio(data, segmentation_criteria):
    """Divides the loan portfolio into segments based on the specified segmentation criteria."""

    segments = {}
    if not segmentation_criteria:
        segments['All'] = data
        return segments

    segment_name = '_'.join([f'{k}_{"_".join(map(str, v))}' for k, v in segmentation_criteria.items()])
    
    query = ' & '.join([f'({k}.isin({v}))' for k, v in segmentation_criteria.items()])

    try:
        filtered_data = data.query(query)
    except TypeError:
        raise TypeError

    segments[segment_name] = filtered_data
    return segments

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices

def train_beta_regression_model(data, predictor_variables, segment_name):
    """Trains a Beta regression model.

    Args:
        data: DataFrame.
        predictor_variables: List of predictor variable names.
        segment_name: Segment name.

    Returns:
        A fitted Beta regression model object.
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise AttributeError("Input data must be a pandas DataFrame.")

        # Handle missing data by dropping rows with NaN values
        data = data.dropna()

        # Construct the formula for the model
        if predictor_variables:
            formula = 'LGD ~ ' + ' + '.join(predictor_variables)
        else:
            formula = 'LGD ~ 1'  # Model with only an intercept

        # Fit the Beta regression model using GLM
        model = smf.glm(formula=formula, data=data, family=sm.families.Beta(link=sm.genmod.families.links.logit())).fit()
        return model
    except Exception as e:
        raise ValueError(f"Error during model training: {e}")

import pandas as pd

def predict_lgd(model, data):
    """Predicts LGD values using the trained model."""
    try:
        predictions = model.predict(data)
        if isinstance(predictions, list):
            predictions = pd.Series(predictions)
        elif not isinstance(predictions, pd.Series):
            predictions = pd.Series(predictions)
        return predictions
    except Exception as e:
        raise e

import pandas as pd

def apply_regulatory_floor(lgd_predictions, floor):
    """Applies a regulatory floor to LGD predictions."""
    return lgd_predictions.clip(lower=floor)

def calculate_pseudo_r_squared(model):
                """    Calculates McFadden's Pseudo-R² for the fitted Beta regression model.
Arguments:
model: The fitted Beta regression model.
Output:
The calculated Pseudo-R² value.
                """

                return 1 - (model.llf / model.llnull)

import matplotlib.pyplot as plt
import numpy as np

def generate_calibration_plot(predicted_lgd, actual_lgd, n_bins):
    """Generates a calibration plot."""
    if len(predicted_lgd) == 0 or len(actual_lgd) == 0:
        raise ValueError("Input arrays cannot be empty.")
    if len(predicted_lgd) != len(actual_lgd):
        raise ValueError("Input arrays must have the same length.")
    if n_bins <= 0:
        raise ValueError("Number of bins must be greater than 0.")

    indices = np.argsort(predicted_lgd)
    predicted_lgd = predicted_lgd[indices]
    actual_lgd = actual_lgd[indices]

    bins = np.linspace(0, len(predicted_lgd), n_bins + 1, dtype=int)
    bin_means_predicted = []
    bin_means_actual = []

    for i in range(n_bins):
        start = bins[i]
        end = bins[i+1]
        if start == end:
            bin_means_predicted.append(np.nan)
            bin_means_actual.append(np.nan)
        else:
            bin_means_predicted.append(np.mean(predicted_lgd[start:end]))
            bin_means_actual.append(np.mean(actual_lgd[start:end]))

    fig, ax = plt.subplots()
    ax.plot(bin_means_predicted, bin_means_actual, marker='o', linestyle='-', label='Calibration Curve')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')

    ax.set_xlabel('Mean Predicted LGD')
    ax.set_ylabel('Mean Actual LGD')
    ax.set_title('Calibration Plot')
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return fig

def calculate_mean_absolute_error(predicted_lgd, actual_lgd):
                """Calculates the Mean Absolute Error (MAE)."""
                absolute_errors = [abs(p - a) for p, a in zip(predicted_lgd, actual_lgd)]
                return sum(absolute_errors) / len(absolute_errors) if absolute_errors else 0.0