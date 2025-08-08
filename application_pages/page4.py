import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib # For saving/loading models

def temporal_split(df, train_size=0.8):
    """
    Splits the data into training and out-of-time (OOT) samples based on time (default_quarter).
    Arguments:
        df: Pandas DataFrame containing the loan data with 'default_quarter_str'.
        train_size: Fraction of data to use for training (based on quarters).
    Output:
        Tuple of two Pandas DataFrames: training and OOT samples.
    """
    if 'default_quarter_str' not in df.columns:
        st.error("Column 'default_quarter_str' not found for temporal split. Ensure Feature Engineering is done.")
        return pd.DataFrame(), pd.DataFrame()

    # Sort by default_quarter_str to ensure temporal order
    df_sorted = df.sort_values(by='default_quarter_str').reset_index(drop=True)

    # Determine the split point based on the number of unique quarters
    unique_quarters = df_sorted['default_quarter_str'].unique()
    split_point_idx = int(len(unique_quarters) * train_size)

    # Get the quarter at the split point
    split_quarter = unique_quarters[split_point_idx]

    train_df = df_sorted[df_sorted['default_quarter_str'] < split_quarter].copy()
    oot_df = df_sorted[df_sorted['default_quarter_str'] >= split_quarter].copy()

    st.info(f"Data split into training ({len(train_df)} rows) and out-of-time ({len(oot_df)} rows) datasets.")
    st.info(f"Training data up to quarter: {unique_quarters[split_point_idx-1] if split_point_idx > 0 else 'N/A'}")
    st.info(f"OOT data from quarter: {split_quarter}")
    return train_df, oot_df

def fit_beta_regression(X_train, y_train):
    """
    Fits a Beta regression model to the training data using statsmodels GLM.
    Arguments:
        X_train: Training features.
        y_train: Training target variable (LGD), must be between 0 and 1.
    Output:
        Trained Beta regression model (statsmodels GLM object).
    """
    # Beta regression requires target variable to be strictly between 0 and 1.
    # Add a small epsilon to 0 and subtract from 1 to handle boundary values.
    epsilon = 1e-6
    y_train_adjusted = y_train.clip(lower=epsilon, upper=1 - epsilon)

    # Add a constant for the intercept
    X_train_sm = sm.add_constant(X_train)

    try:
        # Using GLM with Beta family and logit link function
        model = sm.GLM(y_train_adjusted, X_train_sm, family=sm.families.Beta()).fit()
        st.success("Beta regression model trained successfully!")
        st.markdown(r"\\( \text{The Beta regression model estimates the mean of LGD (\(\mu\)) using a logit link function:}\ \text{logit}(\mu) = \ln\left(\frac{\mu}{1-\mu}\right) = X\beta\\)")
        st.subheader("Model Summary (Coefficients and p-values)")
        st.text(model.summary().as_text())
        return model
    except Exception as e:
        st.error(f"Error fitting Beta regression model: {e}")
        return None

def predict_beta(model, X):
    """
    Predicts LGD values using the trained Beta regression model.
    Arguments:
        model: Trained Beta regression model (statsmodels GLM object).
        X: Input features for prediction.
    Output:
        Predicted LGD values (numpy array).
    """
    if model is None:
        return np.array([])
    
    # Add a constant for the intercept to the prediction data
    X_sm = sm.add_constant(X, has_constant='add') # Use has_constant='add' if X might already have it

    try:
        predictions = model.predict(X_sm)
        return predictions
    except Exception as e:
        st.error(f"Error making predictions: {e}")
        return np.array([])

def apply_lgd_floor(lgd_predictions, floor):
    """
    Applies an LGD floor to the predicted LGD values.
    Arguments:
        lgd_predictions: Predicted LGD values.
        floor: Minimum LGD value.
    Output:
        LGD values with floor applied.
    """
    if lgd_predictions is None or len(lgd_predictions) == 0:
        return np.array([])
    
    floored_predictions = np.maximum(lgd_predictions, floor)
    st.info(f"Applied LGD floor of {floor} to predictions.")
    return floored_predictions

def plot_predicted_vs_actual(y_actual, y_predicted, title="Predicted vs. Actual LGD"):
    """
    Generates a scatter plot of predicted vs. actual LGD with a 45-degree line.
    """
    df_plot = pd.DataFrame({'Actual LGD': y_actual, 'Predicted LGD': y_predicted})
    
    fig = px.scatter(df_plot, x='Actual LGD', y='Predicted LGD',
                     title=title, opacity=0.6, trendline="ols",
                     labels={"Actual LGD": "Actual LGD", "Predicted LGD": "Predicted LGD"},
                     template="plotly_white")
    
    # Add 45-degree line (perfect prediction)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfect Prediction',
                             line=dict(color='red', dash='dash')))
    
    fig.update_layout(xaxis_title="Actual LGD", yaxis_title="Predicted LGD",
                      xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
    return fig

def plot_calibration_curve(y_actual, y_predicted, n_bins=10, title="LGD Calibration Curve"):
    """
    Generates a calibration curve for LGD predictions.
    """
    df_calib = pd.DataFrame({'Actual': y_actual, 'Predicted': y_predicted})
    
    # Create bins for predicted values
    df_calib['Bin'] = pd.cut(df_calib['Predicted'], bins=n_bins, labels=False, include_lowest=True)
    
    # Calculate mean actual and mean predicted for each bin
    calibration_data = df_calib.groupby('Bin').agg(
        mean_predicted=('Predicted', 'mean'),
        mean_actual=('Actual', 'mean')
    ).reset_index()
    
    fig = px.line(calibration_data, x='mean_predicted', y='mean_actual',
                  title=title, markers=True,
                  labels={"mean_predicted": "Mean Predicted LGD", "mean_actual": "Mean Actual LGD"},
                  template="plotly_white")
    
    # Add perfect calibration line
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Perfect Calibration',
                             line=dict(color='red', dash='dash')))
    
    fig.update_layout(xaxis_title="Mean Predicted LGD (in bin)", yaxis_title="Mean Actual LGD (in bin)",
                      xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]))
    return fig

def plot_residuals_vs_fitted(y_actual, y_predicted, title="Residuals vs. Fitted Values"):
    """
    Generates a scatter plot of residuals vs. fitted (predicted) values.
    """
    residuals = y_actual - y_predicted
    df_plot = pd.DataFrame({'Fitted Values': y_predicted, 'Residuals': residuals})
    
    fig = px.scatter(df_plot, x='Fitted Values', y='Residuals',
                     title=title, opacity=0.6,
                     labels={"Fitted Values": "Predicted LGD", "Residuals": "Residuals"},
                     template="plotly_white")
    
    # Add horizontal line at 0 for residuals
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Zero Residuals", annotation_position="bottom right")
    
    fig.update_layout(xaxis_title="Predicted LGD", yaxis_title="Residuals")
    return fig


def run_page4():
    st.header("TTC Model Building")
    st.markdown("""
    This section focuses on building a Through-The-Cycle (TTC) Loss Given Default (LGD) model.
    A TTC model estimates the average LGD over an entire economic cycle, aiming for stability rather than responsiveness to short-term economic fluctuations.

    We will use a Beta regression model, which is suitable for modeling variables (like LGD) that are bounded between 0 and 1.

    ### Steps:
    1.  **Select Features:** Choose the independent variables (features) to train the model.
    2.  **Define LGD Floor:** Set a minimum LGD value to ensure conservative estimates.
    3.  **Train Model:** Fit the Beta regression model on the historical data.
    4.  **Evaluate & Visualize:** Assess model performance using predicted vs. actual plots, calibration curves, and residual plots.
    """)

    if "processed_loan_data" not in st.session_state or st.session_state["processed_loan_data"] is None:
        st.warning("Please go to 'Feature Engineering' page and process the data first.")
        return

    df = st.session_state["processed_loan_data"].copy()

    # Ensure LGD_realised is available and valid
    if 'LGD_realised' not in df.columns or df['LGD_realised'].isnull().any():
        st.error("'LGD_realised' column is missing or contains NaN values. Please ensure Feature Engineering is complete and successful.")
        return

    st.subheader("Model Configuration")

    # Filter out target variable and other non-feature columns for feature selection
    # Identify numerical columns for feature selection, excluding IDs and dates
    exclude_cols = ['id', 'member_id', 'issue_d', 'last_pymnt_d', 'next_pymnt_d', 'last_credit_pull_d', 
                    'loan_status', 'url', 'desc', 'title', 'zip_code', 'addr_state',
                    'default_quarter', 'default_quarter_str', 'issue_d_dt', 'last_pymnt_d_dt',
                    'total_recoveries', 'net_recoveries', 'EAD', 'PV_net_recoveries',
                    'time_to_recovery_years', 'grade', 'sub_grade', 'emp_title', 'emp_length',
                    'cure_status', 'grade_group', 'pymnt_plan', 'policy_code',
                    'hardship_flag', 'hardship_type', 'hardship_reason', 'hardship_status',
                    'deferral_term', 'hardship_amount', 'hardship_start_date', 'hardship_end_date',
                    'payment_plan_start_date', 'hardship_length', 'hardship_dpd',
                    'hardship_loan_status', 'orig_projected_additional_accrued_interest',
                    'hardship_payoff_balance_amount', 'hardship_last_payment_amount',
                    'debt_settlement_flag', 'debt_settlement_flag_date', 'settlement_status',
                    'settlement_date', 'settlement_amount', 'settlement_percentage', 'settlement_term',
                    'LGD_realised' # Exclude target variable
                    ]
    
    available_features = [col for col in df.columns if col not in exclude_cols and df[col].dtype in ['int64', 'float64', 'uint8']]

    selected_features = st.multiselect(
        "Select features for TTC Model Training:",
        options=available_features,
        default=['loan_amnt', 'int_rate', 'term', 'dti', 'annual_inc', 'revol_util', 'emp_length_numeric'] # sensible defaults
    )

    lgd_floor = st.number_input(
        "Enter LGD Floor Value (between 0 and 1):",
        min_value=0.0,
        max_value=1.0,
        value=0.01, # Common to have a small floor
        step=0.005
    )

    st.markdown("---")

    if st.button("Train TTC LGD Model"):
        if not selected_features:
            st.error("Please select at least one feature for model training.")
            return

        if any(col not in df.columns for col in selected_features):
            st.error("One or more selected features are not found in the processed data. Please check Feature Engineering.")
            return

        # Prepare data for modeling
        X = df[selected_features]
        y = df['LGD_realised']
        
        # Handle potential NaN values in selected features (build_features should have handled most)
        # For robustness, drop rows with NaN in X or y here as well.
        data_for_model = pd.concat([X, y], axis=1).dropna()
        if data_for_model.empty:
            st.error("No complete data rows for selected features after dropping NaNs. Adjust feature selection or data preprocessing.")
            st.session_state["ttc_model"] = None
            st.session_state["ttc_predictions"] = None
            return
        
        X_model = data_for_model[selected_features]
        y_model = data_for_model['LGD_realised']

        # Temporal Split (using all data for training if no OOT is specifically required for this page)
        # For a true TTC, we might train on all historical data. But for evaluation, a split is good.
        # Let's use train_test_split as a placeholder for temporal_split here for simplicity for evaluation on this page.
        # The `temporal_split` function is implemented and can be used to emphasize time-based validation.
        train_df, oot_df = temporal_split(data_for_model, train_size=0.8)

        if train_df.empty or oot_df.empty:
            st.warning("Temporal split resulted in empty training or OOT sets. Using all data for training and evaluation for demonstration.")
            X_train, y_train = X_model, y_model
            X_oot, y_oot = X_model, y_model # Use full dataset for 