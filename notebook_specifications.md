
## Jupyter Notebook Specification: LGD Model Development Lab

### 1. Notebook Overview

**Learning Goals:**

*   Understand the components of Loss Given Default (LGD) models, including Through-The-Cycle (TTC) and Point-In-Time (PIT) approaches.
*   Develop skills in data extraction, cleaning, and feature engineering for credit risk modeling.
*   Apply statistical modeling techniques to estimate LGD.
*   Learn how to incorporate macroeconomic factors into LGD models.
*   Evaluate model performance and prepare artifacts for deployment.

**Expected Outcomes:**

Upon completion of this lab, the user will be able to:

*   Calculate Realized LGD from loan-level data.
*   Perform exploratory data analysis to identify key LGD drivers.
*   Build TTC LGD models using appropriate regression techniques.
*   Create PIT overlays to adjust TTC LGDs based on macroeconomic conditions.
*   Assess model fit using relevant metrics and visualizations.
*   Prepare production-ready model artifacts.

**2 Input/Output Expectations:**

*   **Input:** LendingClub loan data (CSV or Parquet format) download it from kaggle from kaggle, macroeconomic data from FRED API. Do not have a load_dataset function.
*   **Output:** Realized LGD values for each loan, TTC LGD model, PIT LGD overlay, model evaluation metrics, visualizations, and saved model artifacts.

**3 Algorithms/Functions:**

*   **Realized LGD Calculation Function:** A function to calculate the realized LGD for each loan based on the EAD, recoveries, and collection costs.
*   **Beta Regression Model Training Function:** A function to train a Beta regression model on the loan data, using loan characteristics as predictors.
*   **PIT Overlay Model Training Function:** A function to train a linear regression model to adjust the TTC LGD based on macroeconomic factors.
*   **Model Evaluation Functions:** Functions to calculate model evaluation metrics, such as pseudo-R-squared, MAE, and calibration plots.

**3.4 Visualizations:**

*   **Histograms and Kernel Density Plots:** Visualize the distribution of `LGD_realized` overall and by `grade_group`.
*   **Box/Violin Plots:** Compare LGD across different loan terms and cure statuses.
*   **Heatmap:** Show the correlations among numeric LGD drivers (e.g., loan size, interest rate, time to default).
*   **Bar Chart:** Illustrate mean LGD by loan grade to justify segmentation decisions.
*   **Violin Plot:** Show the distribution of cured vs. non-cured LGDs.
*   **Scatter Plot:** Compare predicted vs. actual LGD values, with a 45-degree line for reference.
*   **Calibration Curve:** Plot binned mean predicted vs. actual LGD to assess model calibration.
*   **Residuals vs. Fitted Plot:** Examine the residuals of the Beta regression model.
*   **Dual-Axis Line Chart:** Display quarterly average LGD and unemployment rate over time.
*   **Scenario Slider:** Allow the user to visualize the stressed LGD uplift under different macroeconomic scenarios.

### Instructions

*   **Reproducibility:** Set a fixed random seed (e.g., `RANDOM_STATE = 42`) to ensure reproducible results. Store the exact training indices for future reference.
*   **Data Segmentation:** Segment the data based on loan grade (`grade_group`: Prime A–B vs Sub-prime C–G) and cure status (cured vs not cured).
*   **LGD Floor:** Apply a 5% LGD floor as required by regulations.
*   **Model Evaluation:** Evaluate the model on an out-of-time (OOT) sample to assess its performance over time.
*   **Artifacts:** Commit all model artifacts (datasets, models, preprocessors, etc.) to version control with appropriate tags (e.g., `lgd_model_dev_v1`).
*   **Notebook Flow:** Follow the suggested notebook flow:
    *   `00_data_ingestion.ipynb`
    *   `01_feature_engineering.ipynb`
    *   `02_eda_segmentation.ipynb`
    *   `03_ttc_model_build.ipynb`
    *   `04_pit_overlay.ipynb`
    *   `05_model_export.ipynb`
*   **Assumptions:** The analysis assumes the availability of historical loan data with sufficient defaults and recovery information. The effectiveness of the PIT overlay depends on the correlation between macroeconomic factors and LGD.
*   **Constraints:** The model should adhere to regulatory guidelines and internal risk management policies.
*    **Artefact saving:**
    Save the following:
    | Dataset / File                                                                                                               | Purpose                                                                                                               | Exact Fields |
    | ---------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------ |
    | **Saved models & preprocessors** (`/models/lgd_preprocess_v1.pkl`, `lgd_beta_regression_v1.pkl`, `lgd_macro_overlay_v1.pkl`) | Reproduce predictions exactly as deployed in Part 1.                                                                  |              |
    | **Out-of-time (OOT) sample** `oot_2019_defaults.parquet` (30 % hold-out or latest year)                                      | Performance degradation & recalibration need. Must match Part 1 schema (`loan_amnt`, `int_rate`, … , `LGD_realised`). |              |
    | **Quarterly portfolio snapshots** `snap_YYYYQ.csv`                                                                           | Compute Population-Stability-Index (PSI), grade migration, realised default vs predicted LGD by cohort.               |              |
    | **Override log** `overrides.csv`                                                                                             | Assess governance overrides: *obligor\_id, override\_date, model\_grade, final\_grade, reason\_code, approver*.       |              |
    | **Macro forecast scenarios** (FRED API JSON)                                                                                 | Forward-looking PIT validation under baseline, adverse, severely-adverse.                                             |              |
    | **Benchmark LGD study** `industry_LGD_benchmark.xlsx`                                                                        | External plausibility check.                                                                                          |              |

### Functionality Required:

set_seed

fetch_lendingclub_date():
Fetches the dataset from https://resources.lendingclub.com/LoanStats_2018Q4.csv.zip and loads it in a dataframe.

filter_defaults()
assemble_recovery_cashflows()
compute_ead()
pv_cashflows()
compute_realized_lgd()

assign_grade_group()
derive_cure_status()
build_features()

add_default_quarter()
temporal_split()

fit_beta_regression()
predict_beta()
apply_lgd_floor()

aggregate_lgd_by_cohort()
align_macro_with_cohorts()
fit_pit_overlay()
apply_pit_overlay()

mae()
calibration_bins()
residuals_vs_fitted()

plot_lgd_hist_kde()
plot_box_violin()
plot_corr_heatmap()
plot_mean_lgd_by_grade()
plot_pred_vs_actual()
plot_calibration_curve()
plot_quarterly_lgd_vs_unrate()

save_model()
export_oot()
save_quarterly_snapshots()
write_macro_scenarios_json()
