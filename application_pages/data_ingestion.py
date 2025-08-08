"""
import streamlit as st
import pandas as pd

def run_data_ingestion():
    st.header("Data Ingestion")
    st.markdown("Upload the LendingClub dataset in CSV format.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Data loaded successfully!")
            st.dataframe(df)
            st.session_state['data'] = df  # Save the dataframe to session state
        except pd.errors.ParserError as e:
            st.error(f"Error parsing CSV file: {e}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.info("Please upload a CSV file to proceed.")

if 'data' not in st.session_state:
    st.session_state['data'] = None





"""))
