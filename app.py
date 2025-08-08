"""
import streamlit as st
st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()
st.markdown("""
In this lab, we will explore Loss Given Default (LGD) models, a crucial component of credit risk management. LGD represents the expected loss if a borrower defaults on a loan.  This application provides an interactive environment to understand and build LGD models using real-world data and statistical techniques.

We will cover the following key areas:

- **Data Ingestion:** Loading and preprocessing the LendingClub dataset.
- **Feature Engineering:** Creating relevant features from the raw data.
- **EDA and Segmentation:** Exploring the data to identify key drivers of LGD and segment the loan population.
- **TTC Model Building:** Developing a Through-The-Cycle (TTC) LGD model using regression techniques.
- **PIT Overlay:** Incorporating macroeconomic factors to create a Point-In-Time (PIT) LGD model.
- **Model Evaluation:** Evaluating the performance of the LGD models.
- **Model Export:** Saving the trained models and processed data for future use.

**Formulae**

Here are some of the important formulae that will be used in the app

*  $LGD = (Loss \: Amount) / (Exposure \: at \: Default)$
*  $EAD = Outstanding \: Loan \: Amount + Accrued \: Interest - Any \: Pre-Default \: Payments$

"""")
# Your code starts here
page = st.sidebar.selectbox(label="Navigation", options=["Data Ingestion", "Feature Engineering", "EDA and Segmentation", "TTC Model Building", "PIT Overlay", "Model Evaluation", "Model Export"])
if page == "Data Ingestion":
    from application_pages.data_ingestion import run_data_ingestion
    run_data_ingestion()
elif page == "Feature Engineering":
    from application_pages.feature_engineering import run_feature_engineering
    run_feature_engineering()
elif page == "EDA and Segmentation":
    from application_pages.eda_segmentation import run_eda_segmentation
    run_eda_segmentation()
elif page == "TTC Model Building":
    from application_pages.ttc_model_building import run_ttc_model_building
    run_ttc_model_building()
elif page == "PIT Overlay":
    from application_pages.pit_overlay import run_pit_overlay
    run_pit_overlay()
elif page == "Model Evaluation":
    from application_pages.model_evaluation import run_model_evaluation
    run_model_evaluation()
elif page == "Model Export":
    from application_pages.model_export import run_model_export
    run_model_export()
# Your code ends
"""