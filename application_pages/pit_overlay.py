"""
import streamlit as st
import pandas as pd
import plotly.express as px
from application_pages.utils import (
    aggregate_lgd_by_cohort,
    align_macro_with_cohorts,
    fit_pit_overlay
)

def run_pit_overlay():
    st.header("PIT Overlay")
    st.markdown("Incorporate macroeconomic factors for Point-In-Time (PIT) adjustments.")

    if "filtered_data" not in st.session_state or st.session_state["filtered_data"] is None:
        st.warning("Please upload data and perform feature engineering first.")
        return

    if "ttc_model" not in st.session_state:
        st.warning("Please train the TTC model first.")
        return

    df = st.session_state["filtered_data"].copy()
    ttc_model = st.session_state["ttc_model"]
    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]
    lgd_predictions_test_floored = st.session_state["lgd_predictions_test_floored"]
    lgd_floor = st.session_state["lgd_floor"]

    st.subheader("Aggregate LGD by Cohort")
    df_lgd_realised = df.copy() # Use the original dataframe to calculate LGD
    lgd_cohorts = aggregate_lgd_by_cohort(df_lgd_realised)

    st.markdown("### Aggregated LGD by Cohort")
    st.dataframe(lgd_cohorts)

    st.subheader("Macroeconomic Data")
    st.markdown("Imported macroeconomic data (example with unemployment rate)")

    # Example Macro Data
    macro_data = pd.DataFrame({
        "quarter": ["2018Q1", "2018Q2", "2018Q3", "2018Q4", "2019Q1", "2019Q2", "2019Q3", "2019Q4"],
        "unemployment_rate": [4.1, 4.0, 3.9, 3.7, 3.8, 3.6, 3.5, 3.6]
    })
    st.dataframe(macro_data)

    st.subheader("Align Macro Data with LGD Cohorts")
    aligned_data = align_macro_with_cohorts(lgd_cohorts, macro_data)

    st.markdown("### Aligned Data")
    st.dataframe(aligned_data)

    st.subheader("Visualize LGD and Unemployment Rate")
    fig_dual_axis = px.line(aligned_data, x="default_quarter_str", y=["mean_lgd_realised", "unemployment_rate"],
                              title="LGD and Unemployment Rate by Quarter",
                              labels={"value": "Value", "default_quarter_str": "Quarter"})
    fig_dual_axis.update_layout(yaxis_title="LGD", yaxis2=dict(title="Unemployment Rate", overlaying="y", side="right"))
    st.plotly_chart(fig_dual_axis)

    st.subheader("PIT Overlay Model")
    # Basic feature engineering for the PIT overlay
    X_macro = aligned_data[["unemployment_rate"]].dropna()

    # Ensure that the index of y_train and X_macro are aligned before calculating the difference
    y_ttc = lgd_cohorts.set_index(aligned_data["default_quarter_str"])["mean_lgd_realised"].dropna() # TTC LGD
    y_realised = aligned_data.set_index("default_quarter_str")["mean_lgd_realised"].dropna() # Realised LGD

    # Align indices
    common_index = X_macro.index.intersection(y_realised.index)
    X_macro = X_macro.loc[common_index]
    y_realised = y_realised.loc[common_index]

    # Calculate the difference between Realised LGD and TTC LGD
    y_diff = (y_realised - y_ttc.loc[common_index]).dropna()
    # Train the PIT overlay model
    pit_model = fit_pit_overlay(X_macro, y_diff)

    # Scenario Analysis - Stress Testing
    st.subheader("Scenario Analysis")
    unemployment_stress = st.slider("Unemployment Rate (Stress)", min_value=0.0, max_value=15.0, value=6.0, step=0.5)
    stress_data = pd.DataFrame({"unemployment_rate": [unemployment_stress]})
    pit_adjustment = pit_model.predict(stress_data)
    stressed_lgd = y_ttc.mean() + pit_adjustment[0]
    st.markdown(f"Stressed LGD with unemployment rate {unemployment_stress}%: {stressed_lgd:.4f}")






"""