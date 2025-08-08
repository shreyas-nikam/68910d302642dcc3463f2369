
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def temporal_split(df, train_size):
    """Splits data into training and OOT samples based on time."""
    if not 0 <= train_size <= 1:
        raise ValueError("train_size must be between 0 and 1")

    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Sort by a relevant time column if available, otherwise just split by index
    # Assuming 'issue_d' (issue date) is a good proxy for time order if available and parsed
    if 'issue_d' in df.columns:
        df['issue_d'] = pd.to_datetime(df['issue_d'], errors='coerce', format='%b-%Y')
        df = df.sort_values(by='issue_d').reset_index(drop=True)
        df = df.dropna(subset=['issue_d'])
        if df.empty:
            st.warning("No valid dates found for temporal split after dropping NaNs.")
            return pd.DataFrame(), pd.DataFrame()

    num_rows = len(df)
    train_rows = int(num_rows * train_size)

    train_df = df.iloc[:train_rows]
    oot_df = df.iloc[train_rows:]
    return train_df, oot_df

def plot_lgd_hist_kde(df, group_by=None):
    """
    Plots histograms and KDEs of LGD_realized using Plotly, optionally grouped.
    """
    if 'LGD_realized' not in df.columns or df['LGD_realized'].isnull().all():
        st.warning("LGD_realized column is missing or all NaN. Cannot plot distribution.")
        return None

    df_plot = df.copy()
    df_plot['LGD_realized'] = pd.to_numeric(df_plot['LGD_realized'], errors='coerce').dropna()

    if df_plot.empty:
        st.warning("DataFrame is empty after processing LGD_realized. Cannot plot distribution.")
        return None

    if group_by and group_by in df_plot.columns and df_plot[group_by].nunique() > 1:
        fig = px.histogram(df_plot, x="LGD_realized", color=group_by, 
                           marginal="box", # Adds a box plot to show distribution per group
                           barmode="overlay", 
                           histnorm='probability density',
                           title=f"Distribution of Realized LGD by {group_by}")
        fig.update_layout(xaxis_title="Realized LGD", yaxis_title="Density")
    else:
        fig = px.histogram(df_plot, x="LGD_realized", 
                           marginal="box", # Adds a box plot
                           histnorm='probability density',
                           title="Overall Distribution of Realized LGD")
        fig.update_layout(xaxis_title="Realized LGD", yaxis_title="Density")

    return fig

def plot_box_violin(df, category_col, plot_type='violin'):
    """
    Plots box or violin plots of LGD_realized by a categorical column using Plotly.
    """
    if 'LGD_realized' not in df.columns or df['LGD_realized'].isnull().all():
        st.warning("LGD_realized column is missing or all NaN. Cannot plot box/violin.")
        return None
    if category_col not in df.columns:
        st.warning(f"Error: Category column '{category_col}' not found in DataFrame.")
        return None

    df_plot = df.copy()
    df_plot['LGD_realized'] = pd.to_numeric(df_plot['LGD_realized'], errors='coerce').dropna()
    if df_plot.empty:
        st.warning("DataFrame is empty after processing LGD_realized. Cannot plot box/violin.")
        return None

    if plot_type == 'box':
        fig = px.box(df_plot, x=category_col, y='LGD_realized', 
                     title=f"Box Plot of Realized LGD by {category_col}")
    elif plot_type == 'violin':
        fig = px.violin(df_plot, x=category_col, y='LGD_realized', 
                        title=f"Violin Plot of Realized LGD by {category_col}")
    else:
        st.warning("Invalid plot_type. Choose 'box' or 'violin'.")
        return None
    
    fig.update_layout(xaxis_title=category_col, yaxis_title="Realized LGD")
    return fig

def plot_corr_heatmap(df, numeric_cols):
    """
    Plots a correlation heatmap for specified numeric columns using Plotly.
    """
    # Ensure all columns exist and are numeric
    existing_numeric_cols = [col for col in numeric_cols 
                             if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if not existing_numeric_cols:
        st.warning("No valid numeric columns found for heatmap.")
        return None
    
    corr_matrix = df[existing_numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(z=corr_matrix.values,
                                   x=corr_matrix.columns,
                                   y=corr_matrix.index,
                                   colorscale='RdBu', # Red-Blue diverging color scale
                                   zmin=-1, zmax=1))
    
    # Add annotations for correlation values
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            fig.add_annotation(
                x=corr_matrix.columns[j],
                y=corr_matrix.index[i],
                text=f"{corr_matrix.iloc[i, j]:.2f}",
                showarrow=False,
                font=dict(color="black", size=10)
            )

    fig.update_layout(
        title="Correlation Heatmap of Numeric LGD Drivers",
        xaxis_title="Features",
        yaxis_title="Features",
        xaxis_showgrid=False, 
        yaxis_showgrid=False,
        yaxis_autorange='reversed' # To show y-axis in typical matrix order
    )

    return fig

def run_eda_segmentation():
    st.header("EDA and Segmentation")
    st.markdown("""
    This section provides tools for Exploratory Data Analysis (EDA) and data segmentation.
    You can visualize distributions, relationships between variables, and prepare data
    for model building.
    """)

    if 'df_processed' not in st.session_state or st.session_state.df_processed is None or st.session_state.df_processed.empty:
        st.warning("Please process the dataset in the 'Feature Engineering' page first.")
        return

    df_current = st.session_state.df_processed.copy()

    st.markdown("### 1. Temporal Data Split")
    st.markdown("""
    Split your data into training and Out-of-Time (OOT) samples. The OOT sample is crucial
    for robust model validation.
    """)
    train_size = st.slider("Select Training Data Proportion", min_value=0.1, max_value=0.9, value=0.7, step=0.05, key="train_split_slider")

    if st.button("Perform Temporal Split", key="split_button"):
        try:
            train_df, oot_df = temporal_split(df_current, train_size)
            st.session_state.train_df = train_df
            st.session_state.oot_df = oot_df

            st.success(f"Data split successfully! Training set: {len(train_df)} rows, OOT set: {len(oot_df)} rows.")
            st.markdown("#### Training Data Preview:")
            st.dataframe(train_df.head())
            st.markdown("#### OOT Data Preview:")
            st.dataframe(oot_df.head())
        except Exception as e:
            st.error(f"Error performing temporal split: {e}")

    st.markdown("### 2. LGD Realized Distribution")
    st.markdown("""
    Visualize the distribution of `LGD_realized` using histograms and Kernel Density Estimates (KDEs).
    You can optionally group the distribution by a categorical variable.
    """)

    if 'LGD_realized' in df_current.columns:
        categorical_cols = [col for col in df_current.columns if df_current[col].dtype == 'object' or df_current[col].nunique() < 50]
        group_by_option = st.selectbox("Group LGD Distribution by:", 
                                       options=['None'] + sorted(categorical_cols), 
                                       key="lgd_dist_group_by")

        if st.button("Plot LGD Distribution", key="plot_lgd_dist_button"):
            with st.spinner('Generating LGD distribution plot...'):
                fig_lgd_dist = plot_lgd_hist_kde(df_current, group_by=None if group_by_option == 'None' else group_by_option)
                if fig_lgd_dist:
                    st.plotly_chart(fig_lgd_dist)
    else:
        st.warning("'LGD_realized' column not found in the processed DataFrame. Please ensure it's calculated in Feature Engineering.")


    st.markdown("### 3. LGD by Categorical Variables")
    st.markdown("""
    Explore `LGD_realized` across different categories using box or violin plots.
    """)
    if 'LGD_realized' in df_current.columns:
        # Ensure 'grade_group', 'term', 'cure_status' exist, or provide alternatives
        default_categorical_options = []
        if 'grade_group' in df_current.columns: default_categorical_options.append('grade_group')
        if 'term' in df_current.columns: default_categorical_options.append('term')
        if 'cure_status' in df_current.columns: default_categorical_options.append('cure_status')

        available_categorical_cols = [col for col in df_current.columns if df_current[col].dtype == 'object' or df_current[col].nunique() < 50]
        categorical_col_to_plot = st.selectbox("Select Categorical Column:", 
                                               options=default_categorical_options if default_categorical_options else available_categorical_cols,
                                               key="lgd_cat_col")
        plot_type = st.radio("Select Plot Type:", ('box', 'violin'), key="box_violin_radio")

        if categorical_col_to_plot and st.button("Plot LGD by Category", key="plot_lgd_cat_button"):
            with st.spinner('Generating categorical LGD plot...'):
                fig_lgd_cat = plot_box_violin(df_current, categorical_col_to_plot, plot_type)
                if fig_lgd_cat:
                    st.plotly_chart(fig_lgd_cat)
    else:
        st.warning("Cannot plot LGD by categorical variables. 'LGD_realized' column is missing.")

    st.markdown("### 4. Correlation Heatmap of Numeric Drivers")
    st.markdown("""
    Understand the relationships between numeric features using a correlation heatmap.
    """)
    numeric_cols = df_current.select_dtypes(include=np.number).columns.tolist()
    if 'LGD_realized' in numeric_cols:
        numeric_cols.remove('LGD_realized') # Often remove target from direct correlation view initially
    
    selected_numeric_cols = st.multiselect("Select Numeric Columns for Heatmap:", 
                                             options=numeric_cols,
                                             default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols, # Default to first few
                                             key="heatmap_cols")

    if st.button("Plot Correlation Heatmap", key="plot_heatmap_button"):
        if selected_numeric_cols:
            with st.spinner('Generating correlation heatmap...'):
                fig_heatmap = plot_corr_heatmap(df_current, selected_numeric_cols)
                if fig_heatmap:
                    st.plotly_chart(fig_heatmap)
        else:
            st.info("Please select at least one numeric column for the heatmap.")

    st.markdown("---")
    st.markdown("#### Note on State Persistence:")
    st.markdown("""
    The processed DataFrame and split datasets (`train_df`, `oot_df`) are stored in 
    Streamlit's `st.session_state` to ensure data persists as you navigate between pages.
    """)

