def calculate_realized_lgd(ead, recoveries, collection_costs, interest_rate, recovery_times):
                """Calculates the realized Loss Given Default (LGD)."""
                present_value_recoveries = 0
                for recovery, cost, time in zip(recoveries, collection_costs, recovery_times):
                    present_value = (recovery - cost) / (1 + interest_rate)**time
                    present_value_recoveries += present_value

                loss = ead - present_value_recoveries
                lgd = max(0, loss / ead)
                return lgd

import pandas as pd
from betareg import Beta
import numpy as np

def train_beta_regression_model(X_train, y_train):
    """Trains a Beta regression model."""
    if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series):
        raise TypeError("X_train must be a pandas DataFrame and y_train must be a pandas Series.")

    if X_train.empty or y_train.empty:
        raise ValueError("X_train and y_train cannot be empty.")

    if not all((0 < y_train) & (y_train < 1)):
        raise ValueError("All values in y_train must be between 0 and 1.")

    if y_train.nunique() == 1:
        pass

    model = Beta(y_train, X_train).fit()
    return model

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

def train_pit_overlay_model(lgd_ttc, macroeconomic_factors, lgd_realized):
    """Trains a linear regression model for PIT overlay."""

    model = LinearRegression()
    
    # Concatenate LGD_TTC and macroeconomic factors
    X = pd.concat([lgd_ttc, macroeconomic_factors], axis=1)
    X.columns = ['lgd_ttc'] + list(macroeconomic_factors.columns)
    
    # Align indices and handle NaN values
    common_index = X.index.intersection(lgd_realized.index)
    X = X.loc[common_index]
    lgd_realized = lgd_realized.loc[common_index]
    
    X = X.fillna(X.mean())
    lgd_realized = lgd_realized.fillna(lgd_realized.mean())
    
    # Train the model if there is data
    if not X.empty and not lgd_realized.empty:
        model.fit(X, lgd_realized)
    else:
        pass

    return model

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import io
import base64

def calculate_model_evaluation_metrics(y_true, y_pred):
    """Calculates model evaluation metrics and generates a calibration plot."""

    if not isinstance(y_true, pd.Series) or not isinstance(y_pred, pd.Series):
        raise TypeError("y_true and y_pred must be pandas Series.")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    if y_true.empty or y_pred.empty:
        return {'pseudo_r_squared': np.nan, 'mae': np.nan, 'calibration_plot': None}

    # Pseudo R-squared calculation
    numerator = np.sum((y_true - y_pred)**2)
    denominator = np.sum((y_true - np.mean(y_true))**2)
    pseudo_r_squared = 1 - (numerator / denominator)

    # Mean Absolute Error (MAE) calculation
    mae = mean_absolute_error(y_true, y_pred)

    # Calibration plot
    plt.figure()
    calibration_data = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    calibration_data['bucket'] = pd.qcut(calibration_data['y_pred'], q=10, duplicates='drop')
    calibration_curve = calibration_data.groupby('bucket')[['y_true', 'y_pred']].mean()
    plt.plot(calibration_curve['y_pred'], calibration_curve['y_true'], marker='o')
    plt.xlabel('Predicted LGD')
    plt.ylabel('Observed LGD')
    plt.title('Calibration Plot')
    plt.plot([0, 1], [0, 1], 'r--')  # Add a diagonal line for reference
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Convert plot to base64 string
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close()
    calibration_plot_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')


    return {'pseudo_r_squared': pseudo_r_squared, 'mae': mae, 'calibration_plot': calibration_plot_base64}