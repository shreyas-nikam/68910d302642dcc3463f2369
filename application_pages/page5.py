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
    # Example: Quarterly Unemployment Rate and GDP Growth. Data is illustrative.
    # In a real scenario, you'd fetch this from an API or a dataset.
    macro_data = pd.DataFrame({
        'quarter': pd.to_datetime(['2010-03-31', '2010-06-30', '2010-09-30', '2010-12-31',
                                 '2011-03-31', '2011-06-30', '2011-09-30', '2011-12-31',
                                 '2012-03-31', '2012-06-30', '2012-09-30', '2012-12-31',
                                 '2013-03-31', '2013-06-30', '2013-09-30', '2013-12-31',
                                 '2014-03-31', '2014-06-30', '2014-09-30', '2014-12-31',
                                 '2015-03-31', '2015-06-30', '2015-09-30', '2015-12-31',
                                 '2016-03-31', '2016-06-30', '2016-09-30', '2016-12-31',
                                 '2017-03-31', '2017-06-30', '2017-09-30', '2017-12-31',
                                 '2018-03-31', '2018-06-30', '2018-09-30', '2018-12-31']),
        'unemployment_rate': [9.8, 9.6, 9.5, 9.3, 9.0, 9.1, 9.0, 8.5,
                              8.2, 8.2, 7.9, 7.8, 7.7, 7.5, 7.2, 6.7,
                              6.7, 6.2, 5.9, 5.6, 5.5, 5.3, 5.1, 5.0,
                              4.9, 4.9, 5.0, 4.7, 4.5, 4.4, 4.2, 4.1,
                              4.1, 3.9, 3.7, 3.7],
        'gdp_growth': [2.8, 2.5, 2.0, 3.1, 1.6, 1.8, 2.1, 2.5,
                       2.4, 2.2, 1.9, 2.6, 2.0, 2.4, 2.7, 3.0,
                       2.9, 2.5, 2.3, 2.1, 2.0, 2.2, 2.4, 2.6,
                       1.8, 2.1, 2.3, 2.5, 2.7, 2.6, 2.4, 2.8,
                       3.2, 4.2, 3.4, 2.9] # Example GDP growth rates
    })
    macro_data['quarter'] = macro_data['quarter'].dt.to_period('Q') # Convert to quarter period
    macro_data['quarter_str'] = macro_data['quarter'].astype(str)  # String representation for merging
    return macro_data

def aggregate_lgd_by_cohort(df):
    """
    Aggregates LGD by cohort (e.g., loan default quarter).
    Arguments:
        df: Pandas DataFrame containing the loan data with 'default_quarter_str' and 'LGD_realised' and 'ttc_lgd_predicted'.
    Output:
        Pandas DataFrame with aggregated LGD by cohort, including mean LGD and mean TTC LGD.
    """
    if 'default_quarter_str' not in df.columns or 'LGD_realised' not in df.columns or 'ttc_lgd_predicted' not in df.columns:
        st.error("Required columns 'default_quarter_str', 'LGD_realised', or 'ttc_lgd_predicted' not found for aggregation.")
        return pd.DataFrame()
    
    # Group by default_quarter_str and calculate the mean of realized LGD and TTC LGD
    lgd_cohorts = df.groupby('default_quarter_str').agg(
        mean_realized_lgd=('LGD_realised', 'mean'),
        mean_ttc_lgd=('ttc_lgd_predicted', 'mean')
    ).reset_index()
    
    # Sort by quarter for time series plotting
    lgd_cohorts['default_quarter'] = pd.PeriodIndex(lgd_cohorts['default_quarter_str'], freq='Q')
    lgd_cohorts = lgd_cohorts.sort_values(by='default_quarter').reset_index(drop=True)

    st.info("Aggregated mean realized and TTC LGD by default quarter.")
    return lgd_cohorts

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
    st.info("Aligned macroeconomic data with LGD cohorts.")
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
        st.success("PIT Overlay (OLS) model trained successfully!")
        st.markdown(r"$$\text{The PIT overlay model estimates the adjustment to LGD (}\Delta LGD\text{) based on macroeconomic factors:}\ \Delta LGD = \alpha + \beta_1 ME_1 + \beta_2 ME_2 + ... + \epsilon$$")
        st.subheader("PIT Overlay Model Summary")
        st.text(model.summary().as_text())
        return model
    except Exception as e:
        st.error(f"Error fitting PIT overlay model: {e}")
        return None

def predict_pit_overlay(model, X):
    """
    Predicts LGD adjustments using the trained PIT overlay model.
    Arguments:
        model: Trained PIT overlay model (statsmodels OLS object).
        X: Input features (macroeconomic variables) for prediction.
    Output:
        Predicted LGD adjustments (numpy array).
    """
    if model is None:
        return np.array([])

    # Add a constant for the intercept to the prediction data
    X_sm = sm.add_constant(X, has_constant='add')

    try:
        predictions = model.predict(X_sm)
        return predictions
    except Exception as e:
        st.error(f"Error making PIT overlay predictions: {e}")
        return np.array([])

def run_page5():
    st.header("PIT Overlay")
    st.markdown("""
    This section focuses on developing a Point-In-Time (PIT) overlay model to adjust the Through-The-Cycle (TTC) LGD for current macroeconomic conditions.
    PIT adjustments make the LGD estimates more responsive to changes in the economic environment.

    We will:
    1.  **Aggregate LGD:** Group historical LGDs by time cohorts (e.g., quarter).
    2.  **Load Macroeconomic Data:** Incorporate relevant macroeconomic indicators (e.g., unemployment rate, GDP growth).
    3.  **Align Data:** Merge LGD data with macroeconomic data based on time.
    4.  **Train PIT Overlay:** Build a regression model to predict LGD deviations from the TTC level based on macro factors.
    5.  **Apply Stress Scenarios:** Simulate how LGD might change under different economic stresses.
    """)

    if "oot_data" not in st.session_state or st.session_state["oot_data"] is None or \
       "ttc_predictions_oot" not in st.session_state or st.session_state["ttc_predictions_oot"] is None:
        st.warning("Please go to 'TTC Model Building' page and train the TTC model first to generate OOT data and predictions.")
        return

    oot_df = st.session_state["oot_data"].copy()
    ttc_predictions_oot = st.session_state["ttc_predictions_oot"]
    
    # Add TTC predictions to the OOT dataframe for easier aggregation
    oot_df['ttc_lgd_predicted'] = ttc_predictions_oot

    st.subheader("1. Aggregate LGD by Cohort")
    st.markdown("""
    To observe the cyclicality of LGD, we first aggregate the realized LGD and the TTC predicted LGD by default quarter.
    """)
    lgd_cohorts = aggregate_lgd_by_cohort(oot_df)

    if lgd_cohorts.empty:
        st.error("LGD cohort aggregation failed or resulted in empty data.")
        return

    st.dataframe(lgd_cohorts.head())

    st.subheader("2. Load Macroeconomic Data")
    st.markdown("""
    We load historical macroeconomic data to serve as indicators for economic cycles.
    """)
    macro_data = load_macroeconomic_data()
    if macro_data.empty:
        st.error("Failed to load macroeconomic data.")
        return
    st.dataframe(macro_data.head())

    st.subheader("3. Align Macroeconomic Data with LGD Cohorts")
    st.markdown("""
    The aggregated LGD data is then merged with macroeconomic indicators based on their respective quarters.
    """)
    aligned_data = align_macro_with_cohorts(lgd_cohorts, macro_data)

    if aligned_data.empty:
        st.error("Alignment of macroeconomic data with LGD cohorts failed.")
        return

    # Calculate the LGD deviation (Realized LGD - TTC LGD) as the target for PIT overlay
    aligned_data['lgd_deviation'] = aligned_data['mean_realized_lgd'] - aligned_data['mean_ttc_lgd']
    st.dataframe(aligned_data.head())

    st.subheader("4. Train PIT Overlay Model")
    st.markdown("""
    An Ordinary Least Squares (OLS) regression model will be trained to capture the relationship between LGD deviations and selected macroeconomic indicators.
    """)

    # Filter for available macroeconomic features
    available_macro_features = [col for col in macro_data.columns if col not in ['quarter', 'quarter_str']]
    selected_macro_features = st.multiselect(
        "Select Macroeconomic Indicators for PIT Overlay:",
        options=available_macro_features,
        default=['unemployment_rate'] # Default selection
    )

    if st.button("Train PIT Overlay Model"):
        if not selected_macro_features:
            st.error("Please select at least one macroeconomic feature for the PIT overlay model.")
            return
        
        # Ensure selected features are in aligned_data and drop NaNs for model training
        X_pit = aligned_data[selected_macro_features].dropna()
        y_pit = aligned_data['lgd_deviation'][X_pit.index] # Align target with X_pit's index

        if X_pit.empty or y_pit.empty:
            st.error("Data for PIT overlay model is empty after selecting features or dropping NaNs. Ensure sufficient data and selected features have no missing values.")
            st.session_state["pit_overlay_model"] = None
            return

        with st.spinner("Training PIT overlay model..."):
            pit_model = fit_pit_overlay(X_pit, y_pit)
            st.session_state["pit_overlay_model"] = pit_model

            if pit_model is not None:
                # Make predictions (adjustments) using the trained PIT model on the full aligned data
                # This is for visualization, not true out-of-sample prediction yet
                aligned_data['pit_adjustment'] = predict_pit_overlay(pit_model, aligned_data[selected_macro_features].fillna(X_pit.mean())) # Fill NaNs for prediction
                aligned_data['pit_lgd_predicted'] = aligned_data['mean_ttc_lgd'] + aligned_data['pit_adjustment']
                
                st.session_state["aligned_lgd_macro_data"] = aligned_data
                st.success("PIT Overlay model trained and predictions generated.")

                st.subheader("Quarterly LGD and Macroeconomic Indicator")
                st.markdown("""
                This plot shows the trend of mean realized LGD, TTC LGD, and the selected macroeconomic indicator over time. It helps visualize how LGD responds to economic cycles.
                """)

                if selected_macro_features:
                    fig = go.Figure()

                    fig.add_trace(go.Scatter(x=aligned_data['default_quarter_str'], y=aligned_data['mean_realized_lgd'],
                                             mode='lines+markers', name='Mean Realized LGD', yaxis='y'))
                    fig.add_trace(go.Scatter(x=aligned_data['default_quarter_str'], y=aligned_data['mean_ttc_lgd'],
                                             mode='lines+markers', name='Mean TTC LGD', yaxis='y'))
                    fig.add_trace(go.Scatter(x=aligned_data['default_quarter_str'], y=aligned_data[selected_macro_features[0]],
                                             mode='lines+markers', name=selected_macro_features[0], yaxis='y2'))

                    fig.update_layout(
                        title='Quarterly LGDs and Macroeconomic Indicator',
                        xaxis_title='Quarter',
                        yaxis=dict(
                            title='LGD',
                            side='left',
                            range=[0,1]
                        ),
                        yaxis2=dict(
                            title=selected_macro_features[0],
                            overlaying='y',
                            side='right',
                            showgrid=False
                        ),
                        hovermode='x unified',
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("5. Stress Testing Scenario")
                st.markdown("""
                Adjust the slider below to simulate changes in the selected macroeconomic indicator and observe the stressed LGD uplift (change in LGD due to macro factor). This shows the sensitivity of LGD to economic conditions.
                The stressed LGD is calculated as $LGD_{stressed} = LGD_{TTC} + \Delta LGD_{stress}$.
                """)

                if selected_macro_features:
                    # Use the first selected macro feature for the slider for simplicity
                    stress_feature = selected_macro_features[0]
                    min_stress = aligned_data[stress_feature].min()
                    max_stress = aligned_data[stress_feature].max()
                    current_avg = aligned_data[stress_feature].mean()

                    # Allow a wider range for stress testing beyond historical max/min
                    stress_value = st.slider(f"Stress Value for {stress_feature}", 
                                             float(min_stress * 0.8), float(max_stress * 1.2), float(current_avg))
                    
                    # Create a dummy dataframe for prediction with the stressed value
                    # Ensure column names match what the model expects (which includes 'const')
                    stress_X = pd.DataFrame(np.array([[stress_value]]), columns=[stress_feature])
                    stress_X_sm = sm.add_constant(stress_X, has_constant='add')

                    # Predict the stressed adjustment using the PIT model
                    stressed_adjustment = predict_pit_overlay(pit_model, stress_X).iloc[0]

                    # Calculate the stressed LGD uplift relative to the mean of mean_ttc_lgd
                    mean_ttc_lgd_overall = aligned_data['mean_ttc_lgd'].mean()
                    stressed_lgd_uplift = stressed_adjustment # This is the absolute uplift from the regression
                    
                    st.info(f"For a {stress_feature} of {stress_value:.2f}:")
                    st.metric(label="Predicted LGD Adjustment (from regression)", value=f"{stressed_adjustment:.4f}")
                    st.metric(label="Overall Mean TTC LGD", value=f"{mean_ttc_lgd_overall:.4f}")
                    st.metric(label="Stressed LGD (TTC LGD + Adjustment)", value=f"{mean_ttc_lgd_overall + stressed_adjustment:.4f}")

                    st.markdown("""
                    This demonstrates how the PIT overlay allows for forward-looking adjustments to LGD based on expected economic conditions, which is crucial for capital planning and stress testing.
                    """)

                else:
                    st.info("Select macroeconomic features to enable stress testing.")
            else:
                st.error("PIT Overlay Model training failed. Cannot perform stress testing.")
    
    st.markdown("""
    --- 
    **Summary of PIT Overlay:**
    *   PIT overlay models complement TTC models by introducing sensitivity to current and future macroeconomic conditions.
    *   They typically model the deviation of realized LGD from TTC LGD using macroeconomic indicators.
    *   Stress testing capabilities allow for projecting LGD under adverse economic scenarios, a key requirement for regulatory compliance (e.g., CCAR, IFRS 9).
    """)

    st.markdown("### Model Export (PIT Overlay Model)")
    if "pit_overlay_model" in st.session_state and st.session_state["pit_overlay_model"] is not None:
        model_filename = "pit_lgd_overlay_model.joblib"
        joblib.dump(st.session_state["pit_overlay_model"], model_filename)

        with open(model_filename, "rb") as f:
            st.download_button(
                label="Download Trained PIT LGD Overlay Model",
                data=f.read(),
                file_name=model_filename,
                mime="application/octet-stream"
            )
    else:
        st.info("Train the PIT overlay model first to enable download.")
