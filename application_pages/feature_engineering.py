"""
import streamlit as st
import pandas as pd

def run_feature_engineering():
    st.header("Feature Engineering")
    st.markdown("Filter, transform, and create relevant features for LGD modeling.")

    # Check if data is loaded
    if st.session_state["data"] is None:
        st.warning("Please upload data in the Data Ingestion page first.")
        return

    df = st.session_state["data"].copy()

    # Loan Status Filter
    loan_status_options = df["loan_status"].unique().tolist()
    selected_loan_status = st.multiselect("Select Loan Status", loan_status_options, default=["Charged Off"])
    df = df[df["loan_status"].isin(selected_loan_status)]

    # Discount Rate Input
    discount_rate = st.number_input("Enter Discount Rate", min_value=0.0, max_value=1.0, value=0.05, step=0.01)

    st.session_state["filtered_data"] = df  # Save the dataframe to session state

    st.subheader("Filtered Data")
    st.dataframe(df)



"""