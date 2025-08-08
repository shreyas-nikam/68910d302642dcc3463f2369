
import streamlit as st

st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()

st.title("QuLab: LGD Models - Development")
st.divider()

st.markdown("""
In this lab, we will explore the development of Loss Given Default (LGD) models, a critical component in credit risk management. LGD represents the expected loss if a borrower defaults on a loan. We will cover the entire process, from data ingestion and feature engineering to model building, evaluation, and deployment.

This application provides an interactive tool for exploring and visualizing LGD models. We will cover the following topics:

- **Data Ingestion:** Load and display the LendingClub dataset.
- **Feature Engineering:** Filter, transform, and create relevant features.
- **EDA and Segmentation:** Explore the data and identify potential segments.
- **TTC Model Building:** Build a Through-The-Cycle (TTC) LGD model.
- **PIT Overlay:** Incorporate macroeconomic factors for Point-In-Time (PIT) adjustments.
- **Model Evaluation:** Evaluate model performance and visualize results.
- **Model Export:** Download saved artifacts.

Throughout this lab, we will emphasize the importance of understanding the underlying data, making informed modeling decisions, and properly evaluating model performance.

**Formulae Used:**

- Realized LGD: $$\text{LGD} = \frac{\text{EAD} - \text{Recoveries} - \text{Collection Costs}}{\text{EAD}}$$
- Present Value of Cashflows: $$PV = \sum_{t=1}^{n} \frac{CF_t}{(1 + r)^t}$$

""")

page = st.sidebar.selectbox(
    label="Navigation",
    options=[
        "Data Ingestion",
        "Feature Engineering",
        "EDA and Segmentation",
        "TTC Model Building",
        "PIT Overlay",
        "Model Evaluation",
        "Model Export",
    ],
)

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
