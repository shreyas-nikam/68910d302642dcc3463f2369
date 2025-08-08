"""import streamlit as st
st.set_page_config(page_title=\"QuLab\", layout=\"wide\")
st.sidebar.image(\"https://www.quantuniversity.com/assets/img/logo5.jpg\")
st.sidebar.divider()
st.title(\"QuLab\")
st.divider()
st.markdown("""
In this lab, we will explore Loss Given Default (LGD) models, covering both Through-The-Cycle (TTC) and Point-In-Time (PIT) approaches. LGD represents the expected loss when a borrower defaults on a loan.

**Learning Objectives:**

- Understand the key components of LGD models.
- Develop data extraction, cleaning, and feature engineering skills for credit risk modeling.
- Apply statistical modeling techniques to estimate LGD.
- Learn how to incorporate macroeconomic factors into LGD models for PIT adjustments.
- Evaluate model performance and prepare artifacts for deployment.

**Formulae:**

- **LGD (Loss Given Default):**  $\text{LGD} = 1 - \text{Recovery Rate}$
- **Recovery Rate:**  $\frac{\text{Recoveries}}{\text{Exposure at Default (EAD)}}$

"""")

page = st.sidebar.selectbox(label=\"Navigation\", options=[\"Data Ingestion\", \"Feature Engineering\", \"EDA and Segmentation\", \"TTC Model Building\", \"PIT Overlay\", \"Model Evaluation\", \"Model Export\"])
if page == \"Data Ingestion\":
    from application_pages.data_ingestion import run_data_ingestion
    run_data_ingestion()
elif page == \"Feature Engineering\":
    from application_pages.feature_engineering import run_feature_engineering
    run_feature_engineering()
elif page == \"EDA and Segmentation\":
    from application_pages.eda_segmentation import run_eda_segmentation
    run_eda_segmentation()
elif page == \"TTC Model Building\":
    from application_pages.ttc_model_building import run_ttc_model_building
    run_ttc_model_building()
elif page == \"PIT Overlay\":
    from application_pages.pit_overlay import run_pit_overlay
    run_pit_overlay()
elif page == \"Model Evaluation\":
    from application_pages.model_evaluation import run_model_evaluation
    run_model_evaluation()
elif page == \"Model Export\":
    from application_pages.model_export import run_model_export
    run_model_export()
"""