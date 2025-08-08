
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families

# Helper functions for TTC Model Building
def fit_beta_regression(X_train, y_train):
    """
    Fits a Beta regression model using statsmodels.GLM with a Beta distribution.
    Handles y_train values that are exactly 0 or 1 by transforming them slightly.
    """
    if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series):
        raise TypeError("X_train must be a DataFrame and y_train must be a Series.")

    if X_train.empty or y_train.empty:
        st.warning("Training data is empty. Cannot fit Beta regression.")
        return None

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same length.")

    # Transform y_train to be strictly between 0 and 1 for Beta regression
    # A common approach is to use (y * (n - 1) + 0.5) / n
    # where n is the number of observations.
    n = len(y_train)
    epsilon = 1e-5 # Small epsilon to avoid exact 0 or 1
    transformed_y_train = y_train.apply(lambda x: np.clip(x, epsilon, 1 - epsilon))
    # Alternatively, for values exactly 0 or 1, a more robust transformation
    # transformed_y_train = (y_train * (n - 1) + 0.5) / n
    # transformed_y_train = transformed_y_train.clip(lower=epsilon, upper=1-epsilon)

    try:
        # Add a constant to the features for the intercept term
        X_train_const = sm.add_constant(X_train, prepend=False, has_constant='add')

        # Fit the GLM with Beta distribution (requires link function compatible with (0,1))
        # Default link for Beta is logit
        model = GLM(transformed_y_train, X_train_const, family=families.Beta()).fit()
        return model
    except Exception as e:
        st.error(f"Error fitting Beta regression model: {e}")
        return None

def predict_beta(model, X):
    """Predicts LGD values using the trained Beta regression model."""
    if model is None:
        st.warning("Beta regression model is not trained. Cannot make predictions.")
        return np.array([])

    if X.empty:
        st.warning("Input features for prediction are empty.")
        return np.array([])

    try:
        # Add a constant to the features for prediction
        X_const = sm.add_constant(X, prepend=False, has_constant='add')
        predictions = model.predict(X_const)
        return predictions
    except Exception as e:
        st.error(f"Error during Beta model prediction: {e}")
        return np.array([])

def apply_lgd_floor(lgd_predictions, floor):
    """Applies an LGD floor to the predicted LGD values."""
    if not isinstance(lgd_predictions, (np.ndarray, list, pd.Series)):
        raise TypeError("LGD predictions must be an array-like object.")
    if not isinstance(floor, (int, float)):
        raise TypeError("Floor value must be numeric.")

    return np.array([max(x, floor) for x in lgd_predictions])

def plot_predicted_vs_actual(y_true, y_pred, title="Predicted vs. Actual LGD"):
    """Plots predicted vs. actual LGD values with a 45-degree line using Plotly."""
    if len(y_true) == 0 or len(y_pred) == 0:
        st.warning("No data to plot for Predicted vs. Actual.")
        return None

    max_val = max(y_true.max(), y_pred.max()) if isinstance(y_true, pd.Series) else max(max(y_true), max(y_pred))
    min_val = min(y_true.min(), y_pred.min()) if isinstance(y_true, pd.Series) else min(min(y_true), min(y_pred))

    fig = px.scatter(x=y_pred, y=y_true, 
                     labels={'x': 'Predicted LGD', 'y': 'Actual LGD'},
                     title=title, 
                     opacity=0.6)
    fig.add_shape(type="line",
                  x0=min_val, y0=min_val, x1=max_val, y1=max_val,
                  line=dict(color="Red", width=2, dash="dash"),
                  name="45-degree line")
    fig.update_layout(showlegend=False) # Hide shape from legend
    return fig

def plot_calibration_curve(calibration_df, title="LGD Calibration Curve"):
    """Plots a calibration curve using Plotly."""
    if calibration_df.empty or 'mean_predicted_lgd' not in calibration_df.columns or 'mean_actual_lgd' not in calibration_df.columns:
        st.warning("Calibration data is missing or malformed. Cannot plot calibration curve.")
        return None

    fig = px.line(calibration_df, x="mean_predicted_lgd", y="mean_actual_lgd", 
                  markers=True, 
                  title=title,
                  labels={'mean_predicted_lgd': 'Mean Predicted LGD (Binned)', 
                            'mean_actual_lgd': 'Mean Actual LGD (Binned)'})
    
    # Add a perfect calibration line (y=x)
    max_val = max(calibration_df['mean_predicted_lgd'].max(), calibration_df['mean_actual_lgd'].max())
    min_val = min(calibration_df['mean_predicted_lgd'].min(), calibration_df['mean_actual_lgd'].min())
    
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                             mode='lines', name='Perfect Calibration', 
                             line=dict(color='red', dash='dash')))
    fig.update_layout(showlegend=True)
    return fig

def plot_residuals_vs_fitted(residuals, fitted_values, title="Residuals vs. Fitted Values"):
    """Plots residuals vs. fitted values using Plotly."""
    if len(residuals) == 0 or len(fitted_values) == 0:
        st.warning("No data to plot for Residuals vs. Fitted.")
        return None
    
    fig = px.scatter(x=fitted_values, y=residuals, 
                     labels={'x': 'Fitted Values', 'y': 'Residuals'},
                     title=title, 
                     opacity=0.6)
    fig.add_hline(y=0, line_dash="dash", line_color="red", name="Zero Residuals")
    fig.update_layout(showlegend=False)
    return fig

def run_ttc_model_building():
    st.header("TTC Model Building")
    st.markdown("""
    This section focuses on building a Through-The-Cycle (TTC) LGD model.
    We will use a Beta Regression model, which is suitable for modeling variables
    that are bounded between 0 and 1, like LGD.
    """)

    if 'train_df' not in st.session_state or st.session_state.train_df is None or st.session_state.train_df.empty:
        st.warning("Please perform data splitting in the 'EDA and Segmentation' page first to get training data.")
        return
    
    train_df = st.session_state.train_df.copy()

    st.markdown("### 1. Select Features for TTC Model")
    st.markdown("""
    Choose the independent variables (features) to train your LGD model. 
    Ensure `LGD_realized` is available as the target variable.
    """)

    available_features = [col for col in train_df.columns if col not in ['LGD_realized', 'loan_id', 'issue_d', 'loan_status', 'recovery_amount', 'collection_costs', 'EAD', 'cashflow', 'time', 'present_value', 'grade', 'grade_group', 'cure_status']]
    
    default_features = []
    # Pre-select some common numeric features if they exist and are numeric
    for f in ['loan_amnt', 'int_rate', 'dti', 'annual_inc', 'revol_util', 'loan_size_income_ratio', 'int_rate_squared']:
        if f in available_features and pd.api.types.is_numeric_dtype(train_df[f]):
            default_features.append(f)
    
    selected_features = st.multiselect(
        "Select features for the TTC LGD model (X_train):",
        options=sorted(available_features),
        default=default_features,
        key="ttc_features_multiselect"
    )

    lgd_floor = st.number_input(
        "Set LGD Floor Value:", 
        min_value=0.0, max_value=1.0, value=0.01, step=0.01,
        help="Minimum predicted LGD value to ensure conservative estimates.",
        key="lgd_floor_input"
    )

    if not selected_features:
        st.warning("Please select at least one feature to build the model.")
        return
    
    if 'LGD_realized' not in train_df.columns:
        st.error("Missing 'LGD_realized' column. Cannot build LGD model.")
        return

    X_train = train_df[selected_features]
    y_train = train_df['LGD_realized']

    # Drop rows with NaN in selected features or target
    initial_rows = len(X_train)
    combined_data = pd.concat([X_train, y_train], axis=1).dropna()
    X_train = combined_data[selected_features]
    y_train = combined_data['LGD_realized']

    if len(X_train) < initial_rows:
        st.info(f"Dropped {initial_rows - len(X_train)} rows with missing values for selected features or LGD_realized.")

    if X_train.empty or y_train.empty:
        st.warning("Training data is empty after handling missing values. Cannot build model.")
        return

    st.markdown("### 2. Train TTC LGD Model")
    if st.button("Train Beta Regression Model", key="train_ttc_model_button"):
        with st.spinner('Training Beta Regression model...'):
            try:
                ttc_model = fit_beta_regression(X_train, y_train)
                if ttc_model:
                    st.session_state.ttc_model = ttc_model
                    st.success("TTC LGD Model trained successfully!")
                    st.markdown("#### Model Summary:")
                    st.text(ttc_model.summary())

                    # Make predictions on training data for plotting
                    lgd_predictions_train_raw = predict_beta(ttc_model, X_train)
                    lgd_predictions_train_floored = apply_lgd_floor(lgd_predictions_train_raw, lgd_floor)
                    
                    st.session_state.lgd_predictions_train_raw = lgd_predictions_train_raw
                    st.session_state.lgd_predictions_train_floored = lgd_predictions_train_floored
                    st.session_state.y_train_actual = y_train
                    st.session_state.lgd_floor_used = lgd_floor # Store floor for consistency

                else:
                    st.error("Model training failed or returned None.")
            except Exception as e:
                st.error(f"Failed to train TTC LGD model: {e}")

    if 'ttc_model' in st.session_state and st.session_state.ttc_model is not None and \
       'lgd_predictions_train_floored' in st.session_state and st.session_state.y_train_actual is not None:
        
        st.markdown("### 3. Model Visualizations (Training Data)")
        st.markdown("#### Predicted vs. Actual LGD")
        fig_pred_actual = plot_predicted_vs_actual(st.session_state.y_train_actual, st.session_state.lgd_predictions_train_floored)
        if fig_pred_actual:
            st.plotly_chart(fig_pred_actual)

        st.markdown("#### Calibration Curve")
        bins = st.slider("Number of Bins for Calibration Curve:", min_value=5, max_value=20, value=10, step=1, key="calibration_bins_slider")
        # Assuming calibration_bins function from model_evaluation or define here
        # For now, let's define a placeholder or assume it's available.
        # To avoid circular dependency, copy it here or ensure it's imported if placed globally
        from application_pages.model_evaluation import calibration_bins, residuals_vs_fitted # Import if available

        calibration_df = calibration_bins(st.session_state.y_train_actual, st.session_state.lgd_predictions_train_floored, bins=bins)
        fig_calibration = plot_calibration_curve(calibration_df)
        if fig_calibration:
            st.plotly_chart(fig_calibration)

        st.markdown("#### Residuals vs. Fitted Values")
        residuals, fitted_values = residuals_vs_fitted(st.session_state.y_train_actual, st.session_state.lgd_predictions_train_floored)
        fig_residuals = plot_residuals_vs_fitted(residuals, fitted_values)
        if fig_residuals:
            st.plotly_chart(fig_residuals)

        st.markdown("---")
        st.markdown("#### Note on LGD Floor:")
        st.markdown(f"""
        A LGD floor of `{st.session_state.lgd_floor_used:.2f}` has been applied to the predicted LGD values.
        This ensures that even for very low predicted LGDs, there is a minimum assumed loss.
        """)

