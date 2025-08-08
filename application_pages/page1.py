
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


