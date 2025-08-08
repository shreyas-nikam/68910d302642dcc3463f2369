
import streamlit as st
import pandas as pd
import requests
import zipfile
import io
import random

def set_seed(seed):
    """Sets the random seed for reproducibility."""
    if not isinstance(seed, int):
        raise TypeError("Seed must be an integer.")
    random.seed(seed)

def fetch_lendingclub_date():
    """Fetches LendingClub loan data from a specified URL or allows user to upload."""
    st.markdown("### Upload LendingClub Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File successfully loaded!")
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.info("Please upload a CSV file or use the default dataset URL if available.")
        if st.button("Load Default LendingClub Data (2018Q4)"): # Optional: provide a default load button
            url = "https://resources.lendingclub.com/LoanStats_2018Q4.csv.zip"
            try:
                with st.spinner('Fetching data from LendingClub...'):
                    response = requests.get(url)
                    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                    zip_content = response.content

                with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_file:
                    csv_files = zip_file.namelist()
                    if not csv_files:
                        raise Exception("No CSV file found in the zip archive.")
                    csv_file_name = csv_files[0]  # Assuming only one CSV file
                    with zip_file.open(csv_file_name) as csv_file:
                        df = pd.read_csv(csv_file, skiprows=1)
                        df = df.dropna(how='all') # Drop the last row if it's completely empty (summary row)
                st.success("Default LendingClub data loaded successfully!")
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error fetching default data: {e}")
            except pd.errors.ParserError as e:
                st.error(f"CSV parsing error in default data: {e}")
            except Exception as e:
                st.error(f"Error processing default zip file: {e}")
    return df

def run_data_ingestion():
    st.header("Data Ingestion")
    st.markdown("""
    In this section, you can load the LendingClub dataset, which will be used for developing the LGD models.
    You can either upload your own CSV file or load a default dataset provided.
    """)

    # Set seed for reproducibility
    set_seed(42)

    # Use session state to store the dataframe
    if 'df' not in st.session_state:
        st.session_state.df = None

    st.session_state.df = fetch_lendingclub_date()

    if st.session_state.df is not None:
        st.markdown("### Loaded Dataset Preview")
        st.dataframe(st.session_state.df.head())
        st.markdown(f"Dataset has {st.session_state.df.shape[0]} rows and {st.session_state.df.shape[1]} columns.")

    st.markdown("""
    #### Important Notes on Data Loading:
    - The `fetch_lendingclub_date()` function handles both file uploads and fetching a default dataset from a URL.
    - Error handling is implemented to gracefully manage issues during file loading or network requests.
    - The loaded DataFrame is stored in Streamlit's session state to persist across page navigations.
    """)


