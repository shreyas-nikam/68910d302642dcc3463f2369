"""
import streamlit as st
import pandas as pd
import plotly.express as px

def run_eda_segmentation():
    st.header("EDA and Segmentation")
    st.markdown("Explore the data and identify potential segments.")

    # Check if filtered data is available
    if "filtered_data" not in st.session_state or st.session_state["filtered_data"] is None:
        st.warning("Please perform feature engineering first.")
        return

    df = st.session_state["filtered_data"].copy()

    st.subheader("Histograms of LGD_realised")
    fig_overall = px.histogram(df, x="loan_amnt", title="Overall LGD Realised Distribution")
    st.plotly_chart(fig_overall)

    # Feature Selection for Visualization
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    selected_feature = st.selectbox("Select a Feature for Visualization", numeric_columns)

    # Slider for filtering data based on selected numerical column.
    min_value = float(df[selected_feature].min())
    max_value = float(df[selected_feature].max())
    filter_range = st.slider(f"Filter {selected_feature}", min_value, max_value, (min_value, max_value))
    filtered_df = df[(df[selected_feature] >= filter_range[0]) & (df[selected_feature] <= filter_range[1])]

    st.subheader("Visualization")
    fig = px.box(filtered_df, y=selected_feature, title=f"Box plot of {selected_feature}")
    st.plotly_chart(fig)

"""