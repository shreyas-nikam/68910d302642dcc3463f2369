id: 68910d302642dcc3463f2369_documentation
summary: Lab 3.1: LGD Models - Development Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: A Comprehensive Guide to Loss Given Default (LGD) Modeling

This codelab provides a step-by-step guide to building and evaluating Loss Given Default (LGD) models using the QuLab application. LGD is a crucial parameter in credit risk management, representing the potential loss on an exposure in the event of a borrower's default.  This application facilitates the understanding and application of both Through-The-Cycle (TTC) and Point-In-Time (PIT) LGD models.

## Importance of LGD Modeling

Accurate LGD estimation is paramount for:

*   **Regulatory Compliance:**  Meeting requirements set by Basel Accords (e.g., IRB approach).
*   **Capital Adequacy:**  Determining the appropriate level of capital reserves to cover potential losses.
*   **Credit Pricing:**  Informing loan pricing strategies to ensure profitability while managing risk.
*   **Portfolio Management:**  Identifying and mitigating concentrations of risk within a loan portfolio.

## Understanding TTC and PIT LGD Models

This lab focuses on two main LGD modeling approaches:

*   **Through-The-Cycle (TTC) LGD:** Aims to estimate the long-run average LGD, ignoring short-term economic fluctuations.
*   **Point-In-Time (PIT) LGD:** Sensitive to current economic conditions and macroeconomic factors, adjusting the TTC LGD based on the prevailing economic climate.

## Mathematical Foundation
LGD is defined as :
$$LGD = 1 - \frac{Recoveries}{Exposure \ at \ Default \ (EAD)}$$

For more advanced models, especially Beta regression for LGD, the formula for the mean of a Beta-distributed variable $\mu$ is often related to a linear predictor $X\beta$ through a link function, such as the logit link:

$$\text{logit}(\mu) = \ln\left(\frac{\mu}{1-\mu}\right) = X\beta$$

Thus, the predicted LGD $\hat{\mu}$ can be obtained by:

$$\hat{\mu} = \frac{1}{1 + e^{-X\beta}}$$

## Navigation

The application's functionalities are structured across several pages, accessible via the sidebar:

*   **Data Ingestion:** Load and inspect the loan dataset.
*   **Feature Engineering:** Prepare the data by creating relevant features for LGD modeling.
*   **EDA and Segmentation:** Explore the data to understand the distribution of LGD and its relationship with loan characteristics.
*   **TTC Model Building:** Build and evaluate a Through-The-Cycle LGD model.
*   **PIT Overlay:** Develop a Point-In-Time overlay to adjust the TTC LGD based on macroeconomic factors.
*   **Model Evaluation:**  Compare the performance of TTC and PIT models.
*   **Model Export:** Download the trained models and processed data.

## Data Flow Diagram

```mermaid
graph LR
    A[Raw Loan Data (e.g., LendingClub)] --> B(Data Ingestion);
    B --> C(Feature Engineering);
    C --> D{EDA and Segmentation};
    D --> E(TTC Model Building);
    E --> F(PIT Overlay);
    F --> G(Model Evaluation);
    G --> H(Model Export);
```

## Codelab Step: Data Ingestion
Duration: 00:05

This step involves loading the loan dataset, which serves as the foundation for subsequent analysis and modeling.

1.  **Upload Data:**
    *   You can upload a CSV file containing the loan data from your local machine.
    *   Alternatively, you can load a sample LendingClub dataset directly from the application.

2.  **Inspect Data:**
    *   The application displays the first few rows of the loaded dataset, allowing you to verify its contents.
    *   The number of rows and columns is also shown, providing an overview of the dataset's size.
    *   A table with column names is displayed to ensure understanding of features.

```python
import streamlit as st
import pandas as pd
import random
import requests
import zipfile
import io

def set_seed(seed):
    """Sets the random seed for reproducibility."""
    if not isinstance(seed, int):
        raise TypeError("Seed must be an integer.")
    random.seed(seed)

def fetch_lendingclub_date():
    """Fetches LendingClub loan data."""
    url = "https://resources.lendingclub.com/LoanStats_2018Q4.csv.zip"  # Example URL, might need updating
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        zip_content = response.content
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {e}")
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_file:
            csv_files = zip_file.namelist()
            if not csv_files:
                st.error("No CSV file found in the zip archive.")
                return None
            csv_file_name = csv_files[0]  # Assuming only one CSV file
            with zip_file.open(csv_file_name) as csv_file:
                df = pd.read_csv(csv_file, skiprows=1)
                # Drop the last row if it's completely empty (summary row)
                df = df.dropna(how='all')
                return df
    except pd.errors.ParserError as e:
        st.error(f"CSV parsing error: {e}")
        return None
    except Exception as e:
        st.error(f"Error processing zip file: {e}")
        return None


def run_page1():
    set_seed(42)
    st.header("Data Ingestion")
    st.markdown("""
    This section allows you to load the LendingClub dataset, which will be used throughout the LGD model development process.
    You can either upload a CSV file from your local machine or load a sample dataset directly.
    """)

    uploaded_file = st.file_uploader("Upload LendingClub Dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Dataset loaded successfully!")
            st.session_state["loan_data"] = df
            st.write("First 5 rows of the loaded dataset:")
            st.dataframe(df.head())
            st.write(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.info("Please upload a CSV file to proceed or use the sample data option below.")

    if st.button("Load Sample LendingClub Data"):
        with st.spinner("Fetching sample data... This may take a moment."):
            df = fetch_lendingclub_date()
            if df is not None:
                st.success("Sample dataset fetched successfully!")
                st.session_state["loan_data"] = df
                st.write("First 5 rows of the loaded sample dataset:")
                st.dataframe(df.head())
                st.write(f"Sample dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
            else:
                st.error("Failed to fetch sample data.")

    if "loan_data" in st.session_state and st.session_state["loan_data"] is not None:
        st.markdown("""
        #### Dataset Columns Overview
        Below is a list of columns in the loaded dataset. Understanding these columns is crucial for subsequent feature engineering and model building.
        """)
        st.dataframe(pd.DataFrame({'Column Name': st.session_state["loan_data"].columns}))
```

<aside class="positive">
Understanding the dataset's structure and variables is essential for effective feature engineering and model building.
</aside>

## Codelab Step: Feature Engineering
Duration: 00:15

Feature engineering is the process of transforming raw data into features that are suitable for modeling. This step involves several sub-tasks:

1.  **Filter Defaults:**  Isolate loans that have defaulted (e.g., `loan_status = 'Charged Off'`).
2.  **Assemble Cashflows:** Calculate the total and net recoveries from defaulted loans.  Total recoveries might include principal, interest, and fees, while net recoveries subtract collection costs.
3.  **Compute EAD (Exposure at Default):** Determine the outstanding balance at the time of default.
4.  **Compute Realized LGD:** Calculate the actual LGD for each defaulted loan using the formula: $LGD = (EAD - PV \ of \ Net \ Recoveries) / EAD$. Note that LGD is clipped between 0 and 1. The present value is computed by discounting the recoveries.
5.  **Derive Grade Group:**  Categorize loans into prime (A-B) and sub-prime (C-G) groups based on their grade.
6.  **Derive Cure Status:**  Indicate whether a defaulted loan has been "cured" (e.g., fully paid off after default).
7.  **Build Features:** Create additional features, such as numeric representations of employment length and one-hot encoding of categorical features (e.g., home ownership, loan purpose).

```python
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

    st.markdown("")

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
            st.write(df_processed['LGD_realised'].describe())
```

<aside class="negative">
Missing data and incorrect data types can significantly impact model performance.  Ensure proper data cleaning and handling of missing values during feature engineering.  In this application, missing values are dropped, but in real-world scenarios, imputation techniques are preferred.
</aside>

## Codelab Step: EDA and Segmentation
Duration: 00:10

Exploratory Data Analysis (EDA) and data segmentation are crucial for understanding the data's characteristics and identifying patterns related to LGD. This step involves:

1.  **Distribution of Realized LGD:** Visualize the overall distribution of `LGD_realised` using histograms and kernel density plots.
2.  **LGD vs. Loan Characteristics:** Explore the relationship between `LGD_realised` and categorical features (e.g., `grade_group`, `term`, `cure_status`) using box and violin plots.
3.  **Correlation Heatmap:** Generate a heatmap to visualize the correlations between numerical features, including `LGD_realised`.
4.  **Mean LGD by Grade:**  Calculate and visualize the average `LGD_realised` for each loan grade.
5.  **Interactive Data Filtering:**  Use sliders to filter the data based on numerical columns and observe how the distribution of `LGD_realised` changes.

