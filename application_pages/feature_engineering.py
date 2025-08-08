
import streamlit as st
import pandas as pd
import numpy as np
import requests
import zipfile
import io

# Helper functions for Feature Engineering
def filter_defaults(df, selected_status='Charged Off'):
    """Filters the dataset to include only defaulted loans based on selected status."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame.")
    if 'loan_status' not in df.columns:
        raise KeyError("Column 'loan_status' missing in DataFrame for filtering.")
    defaulted_loans = df[df['loan_status'] == selected_status].copy()
    return defaulted_loans

def assemble_recovery_cashflows(df):
    """Assembles recovery cashflows from loan data.
    Assumes 'id' for loan_id, 'recovery_amount', and 'collection_recovery_fee' for collection_costs.
    """
    if df.empty:
        return pd.DataFrame()

    required_columns = ['id', 'recovery_amount', 'collection_recovery_fee']
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' missing in DataFrame. Cannot assemble recovery cashflows.")

    recovery_cashflows = df[required_columns].copy()
    recovery_cashflows.rename(columns={'id': 'loan_id', 'collection_recovery_fee': 'collection_costs'}, inplace=True)
    return recovery_cashflows

def compute_ead(df):
    """Computes the Exposure at Default (EAD) for each loan.
    A simplified EAD is calculated as loan_amnt - total_rec_prncp.
    """
    if not isinstance(df, pd.DataFrame):
        raise AttributeError("Input must be a Pandas DataFrame.")
    
    if 'loan_amnt' not in df.columns:
        raise KeyError("Column 'loan_amnt' missing for EAD calculation.")
    
    if 'total_rec_prncp' in df.columns:
        df['EAD'] = (df['loan_amnt'] - df['total_rec_prncp']).clip(lower=0)
    else:
        st.warning("Column 'total_rec_prncp' not found. Using 'loan_amnt' as EAD proxy.")
        df['EAD'] = df['loan_amnt']

    return df

def pv_cashflows(df, discount_rate):
    """Calculates the present value of cashflows.
    Assumes 'recovery_amount' is the cashflow (CF) and time (t) is 1 for simplicity.
    """
    if not isinstance(discount_rate, (int, float)):
        raise TypeError("Discount rate must be a numeric value.")
    if discount_rate < 0:
        st.warning("Discount rate should ideally be non-negative for this context.")

    df_copy = df.copy()
    
    if 'recovery_amount' in df_copy.columns:
        df_copy['cashflow'] = df_copy['recovery_amount']
    else:
        df_copy['cashflow'] = 0.0 # Default if no recovery amount
        st.warning("'recovery_amount' column not found for PV calculation. Assuming cashflow=0.")

    df_copy['time'] = 1 # Assuming recovery happens at time 1 for PV calculation
    df_copy['present_value'] = df_copy.apply(lambda row: row['cashflow'] / ((1 + discount_rate)**row['time']), axis=1)
    return df_copy

def compute_realized_lgd(df):
    """Computes the realized Loss Given Default (LGD) for each loan."""
    if df.empty:
        st.warning("DataFrame is empty, cannot compute LGD_realized.")
        return df

    required_cols = ['EAD', 'recovery_amount', 'collection_costs']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' missing for LGD_realized calculation. Ensure EAD, recoveries, and collection costs are processed.")

    df['EAD'] = pd.to_numeric(df['EAD'], errors='coerce').fillna(0)
    df['recovery_amount'] = pd.to_numeric(df['recovery_amount'], errors='coerce').fillna(0)
    df['collection_costs'] = pd.to_numeric(df['collection_costs'], errors='coerce').fillna(0)

    df.loc[:, 'LGD_realized'] = (df['EAD'] - df['recovery_amount'] - df['collection_costs']) / df['EAD']
    df.loc[df['EAD'] == 0, 'LGD_realized'] = 0
    df.loc[:, 'LGD_realized'] = df['LGD_realized'].clip(lower=0, upper=1)

    return df

def assign_grade_group(df):
    """Assigns a grade group to each loan based on its grade."""
    def categorize_grade(grade):
        if grade in ['A', 'B']:
            return 'Prime'
        elif grade in ['C', 'D', 'E', 'F', 'G']:
            return 'Sub-prime'
        else:
            return 'Unknown'

    if 'grade' not in df.columns:
        df['grade_group'] = 'Unknown'
        st.warning("Column 'grade' not found. 'grade_group' set to 'Unknown'.")
    else:
        df['grade_group'] = df['grade'].apply(categorize_grade)

    return df

def derive_cure_status(df):
    """Derives cure status for each loan."""
    try:
        if 'loan_status' not in df.columns or 'recovery_amount' not in df.columns:
            raise KeyError("Required columns ('loan_status', 'recovery_amount') not found in DataFrame for cure status derivation.")

        df['cure_status'] = 'not_cured'
        df['recovery_amount'] = pd.to_numeric(df['recovery_amount'], errors='coerce').fillna(0)
        
        df.loc[(df['loan_status'] == 'Fully Paid') & (df['recovery_amount'] > 0), 'cure_status'] = 'cured'
        return df
    except KeyError as e:
        st.error(f"Error deriving cure status: {e}")
        return df

def build_features(df, features):
    """Builds features for the LGD model from the loan data."""
    df_copy = df.copy()

    if not features:
        return df_copy

    for feature in features:
        if feature == 'loan_size_income_ratio':
            if 'loan_amnt' in df_copy.columns and 'annual_inc' in df_copy.columns:
                df_copy['annual_inc'] = pd.to_numeric(df_copy['annual_inc'], errors='coerce').fillna(0)
                df_copy['loan_amnt'] = pd.to_numeric(df_copy['loan_amnt'], errors='coerce').fillna(0)
                df_copy['loan_size_income_ratio'] = np.where(df_copy['annual_inc'] != 0, 
                                                            df_copy['loan_amnt'] / df_copy['annual_inc'], 0)
            else:
                st.warning("Missing 'loan_amnt' or 'annual_inc' for 'loan_size_income_ratio'. Skipping feature.")
                df_copy['loan_size_income_ratio'] = np.nan
        elif feature == 'int_rate_squared':
            if 'int_rate' in df_copy.columns:
                df_copy['int_rate'] = pd.to_numeric(df_copy['int_rate'], errors='coerce').fillna(0)
                df_copy['int_rate_squared'] = df_copy['int_rate']**2
            else:
                st.warning("Missing 'int_rate' for 'int_rate_squared'. Skipping feature.")
                df_copy['int_rate_squared'] = np.nan
        else:
            st.warning(f"Invalid or unsupported feature requested: {feature}. Skipping.")
    return df_copy

def add_default_quarter(df):
    """Adds the default quarter to each loan based on its issue date (as a proxy)."""
    try:
        if 'issue_d' not in df.columns:
            raise KeyError("Column 'issue_d' (proxy for default_date) is missing. Cannot add default quarter.")

        df_copy = df.copy()

        df_copy['issue_d'] = pd.to_datetime(df_copy['issue_d'], format='%b-%Y', errors='coerce')
        df_copy = df_copy.dropna(subset=['issue_d'])

        if df_copy.empty:
            df_copy['default_quarter'] = pd.Series(dtype=str)
            return df_copy

        df_copy['default_quarter'] = df_copy['issue_d'].dt.to_period('Q').astype(str)
        return df_copy
    except ValueError as e:
        st.error(f"Invalid date format in 'issue_d' column: {e}")
        return df
    except TypeError as e:
        st.error(f"Type error in 'issue_d' column: {e}")
        return df
    except KeyError as e:
        st.error(f"Error adding default quarter: {e}")
        return df


def run_feature_engineering():
    st.header("Feature Engineering")
    st.markdown("""
    In this section, you will preprocess the raw LendingClub data to derive features
    essential for LGD model development. This involves filtering defaulted loans,
    assembling recovery cashflows, computing Exposure at Default (EAD), calculating
    realized LGD, and engineering new features.
    """)

    if 'df' not in st.session_state or st.session_state.df is None:
        st.warning("Please load the dataset in the 'Data Ingestion' page first.")
        return

    # Initialize df_processed in session state if not present
    if 'df_processed' not in st.session_state or st.session_state.df_processed is None:
        st.session_state.df_processed = st.session_state.df.copy()
    
    df_current = st.session_state.df_processed.copy() # Start with the current processed df

    st.markdown("### 1. Filter Defaulted Loans")
    st.markdown("""
    We filter the dataset to include only loans that have defaulted. For LGD modeling, we typically focus on 'Charged Off' loans.
    """)

    if 'loan_status' in df_current.columns:
        all_loan_statuses = df_current['loan_status'].unique().tolist()
        # Ensure 'Charged Off' is a valid option if it exists
        default_index = all_loan_statuses.index('Charged Off') if 'Charged Off' in all_loan_statuses else 0
        selected_status = st.selectbox(
            "Select Loan Status to Filter By (for LGD calculation):",
            options=all_loan_statuses,
            index=default_index,
            key="filter_loan_status"
        )
    else:
        st.error("The 'loan_status' column is not found in the dataset. Cannot filter defaults.")
        selected_status = None # Set to None to prevent errors in subsequent calls
    
    if selected_status and st.button(f"Filter by '{selected_status}'", key="filter_button"):
        try:
            df_filtered = filter_defaults(st.session_state.df.copy(), selected_status) # Always filter from original full df
            if df_filtered.empty:
                st.warning(f"No loans found with status '{selected_status}'. Please select another status or check your data.")
            else:
                st.session_state.df_processed = df_filtered
                st.success(f"Filtered dataset to {selected_status} loans. Rows: {len(df_filtered)}")
                st.dataframe(df_filtered.head())
        except Exception as e:
            st.error(f"Error filtering defaults: {e}")

    df_current = st.session_state.df_processed.copy() # Refresh df_current after potential filtering

    if df_current.empty:
        st.warning("The filtered DataFrame is empty. Cannot proceed with further feature engineering steps.")
        return

    st.markdown("### 2. Assemble Recovery Cashflows and Compute EAD")
    st.markdown("""
    Recovery cashflows are the amounts recovered from defaulted loans. Exposure at Default (EAD)
    represents the outstanding amount at the time of default.
    """)
    if st.button("Assemble Recoveries and Compute EAD", key="assemble_ead_button"):
        try:
            df_temp = df_current.copy()
            # Need to ensure 