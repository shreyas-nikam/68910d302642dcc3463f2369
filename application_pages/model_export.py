"""
import streamlit as st
import pandas as pd
import io

def run_model_export():
    st.header("Model Export")
    st.markdown("Download saved model artifacts and datasets.")

    if "ttc_model" not in st.session_state or "X_test" not in st.session_state or "y_test" not in st.session_state:
        st.warning("Please train the TTC model first.")
        return

    ttc_model = st.session_state["ttc_model"]
    X_test = st.session_state["X_test"]
    y_test = st.session_state["y_test"]

    st.subheader("Download Model Artifacts")
    st.markdown("Download the trained TTC model and test dataset.")

    # Create a download button for the model (placeholder)
    st.markdown("#### TTC Model (Placeholder)")
    st.markdown("Model download functionality will be implemented here.")

    # Create a download button for the test dataset
    st.markdown("#### Test Dataset")
    csv_buffer = io.StringIO()
    X_test['lgd_realised'] = y_test  # Combine X_test and y_test for download
    X_test.to_csv(csv_buffer, index=False)
    b64 = base64.b64encode(csv_buffer.getvalue().encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="test_dataset.csv">Download Test Dataset (CSV)</a>'
    st.markdown(href, unsafe_allow_html=True)

import base64

