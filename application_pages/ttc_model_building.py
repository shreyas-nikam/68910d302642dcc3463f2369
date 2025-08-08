"""
import streamlit as st
import pandas as pd
import plotly.express as px
from application_pages.utils import (
    filter_defaults,
    assemble_recovery_cashflows,
    compute_ead,
    pv_cashflows,
    compute_realized_lgd,
    assign_grade_group,
    derive_cure_status,
    build_features,
    add_default_quarter,
    temporal_split,
    fit_beta_regression,
    predict_beta,
    apply_lgd_floor
)
import numpy as np

def run_ttc_model_building():
    st.header("TTC Model Building")
    st.markdown("Build a Through-The-Cycle (TTC) LGD model.")

    if "filtered_data" not in st.session_state or st.session_state["filtered_data"] is None:
        st.warning("Please upload data and perform feature engineering first.")
        return

    df = st.session_state["filtered_data"].copy()

    st.subheader("Data Preparation for TTC Model")
    df_defaults = filter_defaults(df)
    df_recoveries = assemble_recovery_cashflows(df_defaults)
    df_ead = compute_ead(df_recoveries)

    # Discount rate from Feature Engineering page, or default if not set
    discount_rate = st.session_state.get("discount_rate", 0.05)
    df_pv = pv_cashflows(df_ead, discount_rate)
    df_lgd_realised = compute_realized_lgd(df_pv)
    df_lgd_realised = assign_grade_group(df_lgd_realised)
    df_lgd_realised = derive_cure_status(df_lgd_realised)
    df_lgd_realised = add_default_quarter(df_lgd_realised)

    if df_lgd_realised.empty:
        st.warning("No defaulted loans found after filtering. Cannot build TTC model.")
        return

    st.markdown("### Processed Data for TTC Model")
    st.dataframe(df_lgd_realised.head())

    # Feature Selection for TTC Model
    available_features = [
        "loan_amnt", "int_rate", "installment", "annual_inc", "dti", "open_acc",
        "pub_rec", "revol_bal", "revol_util", "total_acc", "recoveries",
        "collection_recovery_fee", "emp_length", "mths_since_last_delinq",
        "mths_since_last_record", "total_pymnt", "total_rec_prncp", "total_rec_int"
    ]
    
    # Filter out features not present in the dataframe
    numeric_cols_in_df = df_lgd_realised.select_dtypes(include=np.number).columns.tolist()
    final_features = [f for f in available_features if f in numeric_cols_in_df and f != "lgd_realised"]
    
    selected_features = st.multiselect(
        "Select features for TTC model training",
        options=final_features,
        default=[f for f in ["loan_amnt", "int_rate", "dti", "annual_inc", "recoveries"] if f in final_features]
    )

    if not selected_features:
        st.warning("Please select at least one feature to train the model.")
        return

    X = build_features(df_lgd_realised, selected_features)
    y = df_lgd_realised["lgd_realised"]

    if X.empty or y.empty:
        st.warning("Feature matrix or target variable is empty. Cannot train model.")
        return

    train_size = st.slider("Training data split (proportion)", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
    X_train, X_test = temporal_split(X.assign(issue_d=df_lgd_realised["issue_d"]), train_size)
    y_train, y_test = temporal_split(pd.DataFrame(y).assign(issue_d=df_lgd_realised["issue_d"]), train_size)

    # Remove the temporary 'issue_d' column from X_train, X_test, y_train, y_test
    X_train = X_train.drop(columns=['issue_d'])
    X_test = X_test.drop(columns=['issue_d'])
    y_train = y_train.drop(columns=['issue_d'])
    y_test = y_test.drop(columns=['issue_d'])

    # Ensure y_train and y_test are series for model fitting
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    st.subheader("Train Beta Regression Model")
    lgd_floor = st.number_input("LGD Floor Value", min_value=0.0, max_value=0.1, value=0.01, step=0.005)

    with st.spinner("Training model..."):
        ttc_model = fit_beta_regression(X_train, y_train)
        lgd_predictions_train = predict_beta(ttc_model, X_train)
        lgd_predictions_test = predict_beta(ttc_model, X_test)

        lgd_predictions_train_floored = apply_lgd_floor(lgd_predictions_train, lgd_floor)
        lgd_predictions_test_floored = apply_lgd_floor(lgd_predictions_test, lgd_floor)
    st.success("Model training complete!")

    st.session_state["ttc_model"] = ttc_model
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test
    st.session_state["lgd_predictions_test_floored"] = lgd_predictions_test_floored
    st.session_state["lgd_floor"] = lgd_floor

    st.subheader("Model Performance (Test Set)")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Predicted vs. Actual LGD")
        plot_df = pd.DataFrame({
            "Actual LGD": y_test,
            "Predicted LGD": lgd_predictions_test_floored
        })
        fig_scatter = px.scatter(plot_df, x="Actual LGD", y="Predicted LGD",
                                title="Predicted vs. Actual LGD (Test Set)",
                                labels={"Actual LGD": "Actual LGD", "Predicted LGD": "Predicted LGD"})
        fig_scatter.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="Red", dash="dash"), name="45-degree line")
        st.plotly_chart(fig_scatter)

    with col2:
        st.markdown("### Calibration Curve")
        # Bin actual vs. predicted for calibration curve
        bins = np.linspace(0, 1, 11) # 10 bins
        plot_df["Predicted_Bin"] = pd.cut(plot_df["Predicted LGD"], bins=bins, include_lowest=True)
        calibration_data = plot_df.groupby("Predicted_Bin").agg(
            mean_predicted=("Predicted LGD", "mean"),
            mean_actual=("Actual LGD", "mean")
        ).reset_index()
        calibration_data["Predicted_Bin_Mid"] = calibration_data["Predicted_Bin"].apply(lambda x: x.mid)

        fig_calibration = px.line(calibration_data, x="Predicted_Bin_Mid", y=["mean_predicted", "mean_actual"],
                                labels={"value": "LGD", "Predicted_Bin_Mid": "Mean Predicted LGD (Binned)"},
                                title="Calibration Curve",
                                line_dash_map={"mean_predicted": "dash", "mean_actual": "solid"})
        fig_calibration.update_layout(legend_title_text='Metric')
        st.plotly_chart(fig_calibration)

    st.subheader("Residuals vs. Fitted Plot")
    residuals = y_test - lgd_predictions_test_floored
    fig_residuals = px.scatter(x=lgd_predictions_test_floored, y=residuals,
                              title="Residuals vs. Fitted (Test Set)",
                              labels={"x": "Fitted LGD", "y": "Residuals"})
    fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_residuals)



