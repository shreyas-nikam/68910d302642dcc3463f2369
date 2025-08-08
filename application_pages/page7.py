import streamlit as st
import pandas as pd
import joblib
import io

def run_page7():
    st.header("Model Export")
    st.markdown("""
    This section allows you to download the trained LGD models (TTC and PIT overlay) 
    and the processed dataset for external use, auditing, or deployment. 
    These artifacts are crucial for integrating the developed models into production systems.

    ### Available Artifacts for Download:
    *   **Processed Loan Data:** The dataset after all feature engineering steps.
    *   **TTC LGD Model:** The trained Through-The-Cycle Beta regression model.
    *   **PIT LGD Overlay Model:** The trained Point-In-Time overlay (OLS) regression model.
    """)

    st.subheader("1. Download Processed Loan Data")
    if "processed_loan_data" in st.session_state and st.session_state["processed_loan_data"] is not None:
        df_processed = st.session_state["processed_loan_data"]
        csv_buffer = io.StringIO()
        df_processed.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Processed Data as CSV",
            data=csv_buffer.getvalue(),
            file_name="processed_loan_data.csv",
            mime="text/csv"
        )
    else:
        st.info("Processed loan data not found. Please complete 'Feature Engineering' first.")

    st.subheader("2. Download Trained TTC LGD Model")
    if "ttc_model" in st.session_state and st.session_state["ttc_model"] is not None:
        model_filename = "ttc_lgd_model.joblib"
        # Save the model to a bytes buffer
        model_buffer = io.BytesIO()
        joblib.dump(st.session_state["ttc_model"], model_buffer)
        model_buffer.seek(0)

        st.download_button(
            label="Download Trained TTC LGD Model",
            data=model_buffer.getvalue(),
            file_name=model_filename,
            mime="application/octet-stream"
        )
    else:
        st.info("TTC LGD model not found. Please train the TTC model in 'TTC Model Building' page.")

    st.subheader("3. Download Trained PIT LGD Overlay Model")
    if "pit_overlay_model" in st.session_state and st.session_state["pit_overlay_model"] is not None:
        model_filename = "pit_lgd_overlay_model.joblib"
        # Save the model to a bytes buffer
        model_buffer = io.BytesIO()
        joblib.dump(st.session_state["pit_overlay_model"], model_buffer)
        model_buffer.seek(0)

        st.download_button(
            label="Download Trained PIT LGD Overlay Model",
            data=model_buffer.getvalue(),
            file_name=model_filename,
            mime="application/octet-stream"
        )
    else:
        st.info("PIT LGD overlay model not found. Please train the PIT overlay model in 'PIT Overlay' page.")

    st.markdown("""
    ---
    **Note on Model Files:**
    The models are saved using `joblib`, a common Python library for serializing and deserializing Python objects. 
    These `.joblib` files can be loaded back into a Python environment using `joblib.load('filename.joblib')`.
    """)
