import streamlit as st
import pandas as pd
import plotly.express as px

def run_page3():
    st.header("EDA and Segmentation")
    st.markdown("""
    This section provides tools for Exploratory Data Analysis (EDA) and data segmentation, helping you understand the distribution of LGD and identify potential relationships between LGD and various loan characteristics.

    We will visualize the distribution of `LGD_realised` and explore how it varies across different segments like `grade_group`, `term`, and `cure_status`.
    """)

    if "processed_loan_data" not in st.session_state or st.session_state["processed_loan_data"] is None:
        st.warning("Please go to 'Feature Engineering' page and process the data first.")
        return

    df = st.session_state["processed_loan_data"].copy()

    st.subheader("Distribution of Realized LGD (`LGD_realised`)")
    st.markdown("""
    Below are histograms and kernel density plots of the `LGD_realised`, both overall and segmented by `grade_group`. This helps in understanding the spread and concentration of losses.
    """)

    # Overall LGD_realised distribution
    fig_overall_lgd = px.histogram(df, x="LGD_realised", nbins=50, 
                                   title="Overall Distribution of Realized LGD",
                                   labels={"LGD_realised": "Realized LGD"},
                                   template="plotly_white")
    fig_overall_lgd.update_traces(marker_color='skyblue', selector=dict(type='histogram'))
    st.plotly_chart(fig_overall_lgd, use_container_width=True)

    # LGD_realised distribution by grade_group
    fig_lgd_by_grade = px.histogram(df, x="LGD_realised", color="grade_group", nbins=50, 
                                    title="Realized LGD Distribution by Grade Group",
                                    labels={"LGD_realised": "Realized LGD", "grade_group": "Grade Group"},
                                    template="plotly_white", barmode="overlay", histnorm="density", opacity=0.7)
    st.plotly_chart(fig_lgd_by_grade, use_container_width=True)

    st.subheader("LGD vs. Loan Characteristics")
    st.markdown("""
    Explore the relationship between `LGD_realised` and other categorical features using box and violin plots.
    """)

    # LGD vs. Term
    fig_lgd_vs_term = px.box(df, x="term", y="LGD_realised", color="term",
                             title="Realized LGD vs. Loan Term",
                             labels={"term": "Loan Term (months)", "LGD_realised": "Realized LGD"},
                             template="plotly_white")
    st.plotly_chart(fig_lgd_vs_term, use_container_width=True)

    # LGD vs. Cure Status
    fig_lgd_vs_cure = px.violin(df, x="cure_status", y="LGD_realised", color="cure_status",
                                title="Realized LGD vs. Cure Status",
                                labels={"cure_status": "Cure Status", "LGD_realised": "Realized LGD"},
                                template="plotly_white")
    st.plotly_chart(fig_lgd_vs_cure, use_container_width=True)

    st.subheader("Correlation Heatmap of Numerical Features")
    st.markdown("""
    A heatmap showing the Pearson correlations among numerical features helps identify multicollinearity and strong relationships with `LGD_realised`.
    """)

    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    if 'LGD_realised' in numerical_cols:
        numerical_cols.remove('LGD_realised') # Remove LGD_realised from features for correlation matrix, as it is the target
    numerical_cols_for_corr = ['LGD_realised'] + numerical_cols # Add LGD_realised to the beginning

    # Filter out columns with all NaN or infinite values before computing correlation
    df_corr = df[numerical_cols_for_corr].copy()
    df_corr = df_corr.dropna(axis=1, how='all')
    df_corr = df_corr.replace([float('inf'), -float('inf')], pd.NA).dropna(axis=1)
    
    if not df_corr.empty:
        corr_matrix = df_corr.corr(numeric_only=True) # Use numeric_only=True
        fig_corr = px.heatmap(corr_matrix, 
                              annotations=True, 
                              title="Pearson Correlation Heatmap of Numerical Features",
                              color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("No numerical columns available for correlation heatmap after cleaning.")

    st.subheader("Mean LGD by Grade")
    st.markdown("""
    This bar chart shows the average `LGD_realised` for each loan grade, providing insights into how credit quality impacts losses.
    """)
    mean_lgd_by_grade = df.groupby('grade')['LGD_realised'].mean().reset_index()
    fig_mean_lgd_grade = px.bar(mean_lgd_by_grade, x="grade", y="LGD_realised",
                                title="Mean Realized LGD by Loan Grade",
                                labels={"grade": "Loan Grade", "LGD_realised": "Mean Realized LGD"},
                                template="plotly_white", color="grade")
    st.plotly_chart(fig_mean_lgd_grade, use_container_width=True)

    st.subheader("Interactive Data Filtering")
    st.markdown("""
    Use the sliders below to filter the data based on numerical columns and observe how the distribution of `LGD_realised` changes.
    """)

    selected_num_col = st.selectbox("Select a numerical column to filter:", 
                                    options=[col for col in df.select_dtypes(include=['number']).columns if col != 'LGD_realised'])

    if selected_num_col:
        min_val, max_val = float(df[selected_num_col].min()), float(df[selected_num_col].max())
        if min_val == max_val:
            st.info(f"Cannot filter on {selected_num_col} as it has a constant value.")
        else:
            filter_range = st.slider(f"Filter by {selected_num_col}", min_val, max_val, (min_val, max_val))
            filtered_df = df[(df[selected_num_col] >= filter_range[0]) & (df[selected_num_col] <= filter_range[1])]

            st.write(f"Displaying distribution for `LGD_realised` based on filtered `{selected_num_col}`.")
            fig_filtered_lgd = px.histogram(filtered_df, x="LGD_realised", nbins=50,
                                            title=f"Distribution of Realized LGD (Filtered by {selected_num_col})",
                                            labels={"LGD_realised": "Realized LGD"},
                                            template="plotly_white")
            st.plotly_chart(fig_filtered_lgd, use_container_width=True)
            st.write(f"Filtered data contains {filtered_df.shape[0]} rows.")

    st.markdown("""
    --- 
    **Summary of EDA Findings:**
    *   The distribution of `LGD_realised` often shows concentrations at 0 (full recovery) and 1 (total loss).
    *   Sub-prime grades (C-G) generally exhibit higher and more variable LGDs compared to Prime grades (A-B).
    *   Longer loan terms (`term`) tend to be associated with higher LGDs, possibly due to increased uncertainty over time.
    *   "Not Cured" loans (defaulted) naturally have higher LGDs than "Cured" loans (paid off or current).
    *   Correlation analysis helps identify features that are strongly related to LGD and could be good predictors for modeling.
    """)


