import streamlit as st
import pandas as pd
import numpy as np
import datetime

# Helper functions for Feature Engineering

def filter_defaults(df):
    """
    Filters the dataset to include only defaulted loans (loan_status is 'Charged Off').
    Arguments:
        df: Pandas DataFrame containing the loan data.
    Output:
        Pandas DataFrame with only defaulted loans.
    """
    if 'loan_status' not in df.columns:
        st.error("Column 'loan_status' not found in the DataFrame.")
        return pd.DataFrame()
    
    # Define default statuses - typically 'Charged Off' for LGD modeling
    default_statuses = ['Charged Off']
    defaulted_df = df[df['loan_status'].isin(default_statuses)].copy()
    st.info(f"Filtered {len(defaulted_df)} defaulted loans (loan_status in {default_statuses}).")
    return defaulted_df

def assemble_recovery_cashflows(df):
    """
    Assembles recovery cashflows from loan data.
    Arguments:
        df: Pandas DataFrame containing the loan data.
    Output:
        Pandas DataFrame with recovery cashflow information.
    """
    if 'total_rec_prncp' not in df.columns or 'total_rec_int' not in df.columns or \
       'total_rec_late_fee' not in df.columns or 'recoveries' not in df.columns or \
       'collection_recovery_fee' not in df.columns:
        st.error("One or more required recovery columns not found. Ensure 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee' exist.")
        return df

    # Total recoveries include principal, interest, late fees, and direct recoveries
    df['total_recoveries'] = df['total_rec_prncp'] + df['total_rec_int'] + \
                             df['total_rec_late_fee'] + df['recoveries']

    # Net recoveries consider the collection costs
    df['net_recoveries'] = df['total_recoveries'] - df['collection_recovery_fee']
    st.info("Assembled 'total_recoveries' and 'net_recoveries' cashflows.")
    return df

def compute_ead(df):
    """
    Computes the Exposure at Default (EAD) for each loan.
    Arguments:
        df: Pandas DataFrame containing the loan data.
    Output:s
        Pandas DataFrame with EAD calculated.
    """
    # EAD is typically funded_amnt for charged-off loans
    # For revolving facilities, it might involve current balance or limit.
    # For this dataset, funded_amnt is a reasonable proxy at default.
    if 'funded_amnt' not in df.columns or 'total_pymnt' not in df.columns:
        st.error("Required columns 'funded_amnt' or 'total_pymnt' for EAD calculation not found.")
        return df
    
    # EAD can be approximated as the outstanding principal at default.
    # Since we are working with charged-off loans, the 'out_prncp' at charge-off date
    # would be the most accurate. If not available, funded_amnt - total_rec_prncp
    # can be a proxy. Here, let's use the remaining principal.
    # LendingClub provides 'out_prncp' which is the outstanding principal balance.
    if 'out_prncp' in df.columns:
        df['EAD'] = df['out_prncp']
    else:
        # Fallback if 'out_prncp' is not directly available or reliable for charged off loans
        # This is a simplification; in real-world, EAD can be complex.
        df['EAD'] = df['funded_amnt'] - df['total_rec_prncp']
        df['EAD'] = df['EAD'].clip(lower=0) # EAD cannot be negative

    st.info("Computed Exposure at Default (EAD).")
    return df

def pv_cashflows(df, discount_rate):
    """
    Calculates the present value of cashflows.
    Arguments:
        df: Pandas DataFrame containing the loan data.
        discount_rate: Discount rate for present value calculation.
    Output:
        Pandas DataFrame with present value of cashflows.
    """
    if 'last_pymnt_d' not in df.columns or 'issue_d' not in df.columns:
        st.error("Required date columns 'last_pymnt_d' or 'issue_d' not found for PV calculation.")
        return df
        
    # Convert date columns to datetime objects
    # Handle potential parsing errors by coercing invalid dates to NaT
    df['issue_d_dt'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
    df['last_pymnt_d_dt'] = pd.to_datetime(df['last_pymnt_d'], format='%b-%Y', errors='coerce')

    # Calculate time to recovery (in years)
    # Assuming recovery happens around last_pymnt_d for simplicity.
    # This is a simplification. Actual recovery cashflows can be spread over time.
    df['time_to_recovery_years'] = (df['last_pymnt_d_dt'] - df['issue_d_dt']).dt.days / 365.25
    df['time_to_recovery_years'] = df['time_to_recovery_years'].fillna(0).clip(lower=0) # Handle NaNs and negatives

    # Convert annual discount rate to a per-period rate (assuming continuous compounding for simplicity)
    # or simple interest if discount_rate is low.
    # For simplicity, using a simple compounding for annual rates over fraction of a year
    # PV = Future_Value / (1 + r)^t
    
    if 'net_recoveries' not in df.columns:
        st.error("Column 'net_recoveries' not found. Please run assemble_recovery_cashflows first.")
        return df

    df['PV_net_recoveries'] = df['net_recoveries'] / ((1 + discount_rate) ** df['time_to_recovery_years'])
    st.info(f"Computed Present Value of net recoveries using a discount rate of {discount_rate}.")
    return df

def compute_realized_lgd(df):
    """
    Computes the realized Loss Given Default (LGD) for each loan.
    Arguments:
        df: Pandas DataFrame containing the loan data with EAD, recoveries and collection costs.
    Output:
        Pandas DataFrame with realized LGD calculated.
    """
    if 'EAD' not in df.columns or 'PV_net_recoveries' not in df.columns:
        st.error("Required columns 'EAD' or 'PV_net_recoveries' for LGD calculation not found. Ensure EAD and PV of cashflows are computed.")
        return df
        
    # LGD = (EAD - PV_net_recoveries) / EAD
    # Handle cases where EAD is zero to prevent division by zero
    df['LGD_realised'] = np.where(df['EAD'] > 0, (df['EAD'] - df['PV_net_recoveries']) / df['EAD'], 1.0)
    
    # LGD must be between 0 and 1
    df['LGD_realised'] = df['LGD_realised'].clip(lower=0.0, upper=1.0)
    st.info("Computed Realized Loss Given Default (LGD_realised).")
    return df

def assign_grade_group(df):
    """
    Assigns a grade group (Prime A-B vs Sub-prime C-G) to each loan based on its grade.
    Arguments:
        df: Pandas DataFrame containing the loan data.
    Output:
        Pandas DataFrame with grade group assigned.
    """
    if 'grade' not in df.columns:
        st.error("Column 'grade' not found for assigning grade groups.")
        return df
        
    df['grade_group'] = df['grade'].apply(lambda x: 'Prime A-B' if x in ['A', 'B'] else 'Sub-prime C-G')
    st.info("Assigned 'grade_group' based on loan grade.")
    return df

def derive_cure_status(df):
    """
    Derives the cure status (cured vs not cured) for each loan.
    For simplicity, 'Cured' means total_rec_prncp is close to funded_amnt,
    and 'Charged Off' indicates 'Not Cured'.
    Arguments:
        df: Pandas DataFrame containing the loan data.
    Output:
        Pandas DataFrame with cure status derived.
    """
    if 'loan_status' not in df.columns or 'total_rec_prncp' not in df.columns or 'funded_amnt' not in df.columns:
        st.error("Required columns 'loan_status', 'total_rec_prncp', or 'funded_amnt' not found for deriving cure status.")
        return df

    # A simplistic approach for cure status:
    # If the loan is 'Charged Off', it's 'Not Cured'.
    # Otherwise, if total principal received is close to funded amount, consider it 'Cured'.
    # This might need more sophisticated business rules in a real scenario.
    
    df['cure_status'] = 'Not Cured' # Default to Not Cured
    
    # Loans that are fully paid or current might be considered "cured"
    # For LGD, we are mostly interested in defaulted loans.
    # So, for charged-off loans, the cure status is 'Not Cured'
    # For loans that were paid off, they would be 'Cured'.
    
    # For this context, we're likely processing only defaulted loans,
    # so 'cure_status' is often 'Not Cured' for these, or 'Cured' if they
    # recovered fully without being charged off.
    # Let's consider 'Fully Paid' as 'Cured' if such loans were included in the input df.
    if 'Fully Paid' in df['loan_status'].unique():
        df.loc[df['loan_status'] == 'Fully Paid', 'cure_status'] = 'Cured'
    
    # If a loan is charged off, it's definitively not cured.
    df.loc[df['loan_status'] == 'Charged Off', 'cure_status'] = 'Not Cured'

    st.info("Derived 'cure_status' for loans.")
    return df

def build_features(df, features):
    """
    Builds features for the LGD model from the loan data.
    Arguments:
        df: Pandas DataFrame containing the loan data.
        features: List of features to build.
    Output:
        Pandas DataFrame with engineered features.
    """
    # Example of feature engineering: handling categorical variables, creating ratios
    processed_df = df.copy()

    # Convert 'term' to numeric (e.g., ' 36 months' -> 36)
    if 'term' in processed_df.columns:
        processed_df['term'] = processed_df['term'].astype(str).str.replace(' months', '').str.strip().astype(int)
        st.info("Converted 'term' to numeric.")

    # Convert 'emp_length' to numeric (e.g., '10+ years' -> 10, '< 1 year' -> 0)
    if 'emp_length' in processed_df.columns:
        emp_length_map = {
            '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
            '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
            '10+ years': 10
        }
        processed_df['emp_length_numeric'] = processed_df['emp_length'].map(emp_length_map).fillna(0) # Fill NaN with 0 or median
        st.info("Converted 'emp_length' to numeric ('emp_length_numeric').")
        features.append('emp_length_numeric') # Add to features list if not already there

    # One-hot encode categorical features like 'home_ownership', 'purpose', 'verification_status'
    # Only if they are in the initial features list or deemed important.
    categorical_features = ['home_ownership', 'purpose', 'verification_status', 'initial_list_status', 'application_type']
    for cat_col in categorical_features:
        if cat_col in processed_df.columns:
            # Handle potential NaN values by filling with a placeholder before one-hot encoding
            processed_df[cat_col] = processed_df[cat_col].fillna('Unknown')
            dummies = pd.get_dummies(processed_df[cat_col], prefix=cat_col, drop_first=True)
            processed_df = pd.concat([processed_df, dummies], axis=1)
            # Remove original categorical column if its one-hot encoded version is used
            if cat_col in features:
                features.remove(cat_col)
            features.extend(dummies.columns.tolist())
            st.info(f"One-hot encoded '{cat_col}'.")

    # Select and return only the specified features and LGD_realised
    final_features = [f for f in features if f in processed_df.columns]
    if 'LGD_realised' in processed_df.columns and 'LGD_realised' not in final_features:
        final_features.append('LGD_realised')
    
    # Ensure all features exist, drop rows with NaN in critical features
    # For simplicity, dropping NaNs, in a real scenario, imputation would be used.
    processed_df = processed_df[final_features].dropna()
    st.info(f"Built features. Final DataFrame shape: {processed_df.shape}")
    return processed_df

def add_default_quarter(df):
    """
    Adds the default quarter to each loan based on its default date.
    Assuming 'issue_d' can be used as a proxy for default date for charged-off loans
    if an explicit default date column is not available or reliable.
    In a real system, 'last_pymnt_d' or a specific 'default_date' would be used.
    Arguments:
        df: Pandas DataFrame containing the loan data.
    Output:
        Pandas DataFrame with default quarter added.
    """
    if 'issue_d' not in df.columns:
        st.error("Column \'issue_d\' not found for adding default quarter.")
        return df

    # Convert 'issue_d' to datetime, handling potential errors
    df['issue_d_dt'] = pd.to_datetime(df['issue_d'], format='%b-%Y', errors='coerce')
    
    # Drop rows where issue_d_dt could not be parsed
    df.dropna(subset=['issue_d_dt'], inplace=True)

    # Extract year and quarter
    df['default_quarter'] = df['issue_d_dt'].dt.to_period('Q')
    df['default_quarter_str'] = df['default_quarter'].astype(str) # For easier plotting/grouping
    st.info("Added 'default_quarter' to the DataFrame.")
    return df

def run_page2():
    st.header("Feature Engineering")
    st.markdown(
    """
    This section is dedicated to preparing the raw loan data for LGD model development.
    It involves filtering for defaulted loans, assembling cashflows, computing Exposure at Default (EAD),
    and deriving the Realized Loss Given Default (LGD_realised).
    Additionally, we will create new features like 'grade_group' and 'cure_status' to enhance model performance.

    ### Key Steps:
    1.  **Filter Defaults:** Isolate loans that have defaulted.
    2.  **Assemble Cashflows:** Consolidate principal, interest, late fees, and recovery amounts.
    3.  **Compute EAD:** Calculate the Exposure at Default.
    4.  **Compute Realized LGD:** Determine the actual LGD for each defaulted loan.
    5.  **Derive Grade Group:** Categorize loans into 'Prime A-B' and 'Sub-prime C-G'.
    6.  **Derive Cure Status:** Identify if a loan has been 'Cured' (e.g., fully paid) or 'Not Cured' (charged off).
    7.  **Build Features:** Engineer additional features and handle categorical variables for modeling.

    All intermediate and final processed dataframes will be saved in Streamlit's session state for use in subsequent pages.
    """
    )

    if "loan_data" not in st.session_state or st.session_state["loan_data"] is None:
        st.warning("Please go to \'Data Ingestion\' page and load the dataset first.")
        return

    df_original = st.session_state["loan_data"].copy()

    st.subheader("Configuration for Feature Engineering")

    selected_loan_statuses = st.multiselect(
        "Select Loan Statuses to consider as Defaulted:",
        options=df_original['loan_status'].unique(),
        default=['Charged Off']
    )

    discount_rate = st.number_input(
        "Enter Discount Rate for Present Value Calculation (e.g., 0.05 for 5%):",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.01
    )

    st.markdown("---")

    if st.button("Perform Feature Engineering"):
        if not selected_loan_statuses:
            st.error("Please select at least one loan status to define as 'defaulted'.")
            return

        with st.spinner("Processing data..."):
            df = df_original.copy()

            st.info(f"Initial dataset shape: {df.shape}")

            # Step 1: Filter defaults
            df = df[df['loan_status'].isin(selected_loan_statuses)].copy()
            if df.empty:
                st.warning("No defaulted loans found with the selected statuses. Adjust filters or check data.")
                st.session_state["processed_loan_data"] = None
                return

            st.write(f"After filtering for selected loan statuses: {df.shape[0]} rows.")

            # Step 2: Assemble Recovery Cashflows
            df = assemble_recovery_cashflows(df)

            # Step 3: Compute EAD
            df = compute_ead(df)

            # Step 4: Calculate Present Value of Cashflows
            df = pv_cashflows(df, discount_rate)

            # Step 5: Compute Realized LGD
            df = compute_realized_lgd(df)

            # Step 6: Assign Grade Group
            df = assign_grade_group(df)

            # Step 7: Derive Cure Status
            df = derive_cure_status(df)

            # Step 8: Add Default Quarter for time-based analysis
            df = add_default_quarter(df)

            # Step 9: Build Features (select and transform final features for modeling)
            # Define a list of potential features. These might be refined later in TTC model building page.
            # Select numerical features that are likely to be useful, and relevant categorical ones for one-hot encoding
            potential_features = [
                'loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'dti',
                'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc',
                'out_prncp', 'total_pymnt', 'total_rec_prncp', 'total_rec_int',
                'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt',
                'collections_12_mths_ex_med', 'acc_now_delinq', 'tot_coll_amt',
                'tot_cur_bal', 'total_rev_hi_lim',
                # Categorical features that will be one-hot encoded by build_features
                'grade', # Keep grade to check grade_group assignment, but one-hot encoding may happen based on usage.
                'home_ownership', 'purpose', 'verification_status', 'initial_list_status',
                'application_type'
            ]
            
            # Filter features that actually exist in the DataFrame
            existing_features = [col for col in potential_features if col in df.columns]

            # Ensure 'grade' is explicitly included in features if grade_group is used in EDA.
            if 'grade' not in existing_features and 'grade' in df.columns:
                existing_features.append('grade')
            
            # Ensure 'emp_length' is present if 'emp_length_numeric' is desired
            if 'emp_length' in df.columns and 'emp_length' not in existing_features:
                 existing_features.append('emp_length')

            df_processed = build_features(df, existing_features)
            
            # Ensure LGD_realised is between 0 and 1 before saving
            df_processed['LGD_realised'] = df_processed['LGD_realised'].clip(lower=0.0, upper=1.0)

            st.session_state["processed_loan_data"] = df_processed
            st.session_state["feature_engineering_done"] = True
            st.success("Feature Engineering Completed!")

            st.subheader("Preview of Processed Data")
            st.write("First 5 rows of the data after feature engineering:")
            st.dataframe(df_processed.head())
            st.write(f"Processed dataset contains {df_processed.shape[0]} rows and {df_processed.shape[1]} columns.")
            
            # Display summary statistics of LGD_realised
            st.subheader("Summary Statistics for Realized LGD")
            st.write(df_processed['LGD_realised'].describe()))