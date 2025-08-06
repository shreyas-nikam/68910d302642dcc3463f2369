
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

### 2. Mathematical and Theoretical Foundations

**2.1 Realized LGD Calculation:**

The realized LGD is calculated for each defaulted loan as:

$$LGD_{realized} = \frac{EAD - PV(Recoveries) - PV(Collection\, Costs)}{EAD}$$

Where:

*   $LGD_{realized}$ is the realized Loss Given Default (a value between 0 and 1).
*   $EAD$ is the Exposure at Default, representing the outstanding principal at the time of default.
*   $PV(Recoveries)$ is the present value of all recoveries associated with the defaulted loan, discounted to the default date using the loan's effective interest rate.
*   $PV(Collection\, Costs)$ is the present value of all collection costs associated with the defaulted loan, discounted to the default date using the loan's effective interest rate.

The present value of a recovery payment at time $t$ is given by:

$$PV(Recovery_t) = \frac{Recovery_t}{(1 + r)^t}$$

where:

*   $Recovery_t$ is the recovery amount at time $t$ after default.
*   $r$ is the loan's effective interest rate.
*   $t$ is the time (in years) from the default date to the recovery payment date.

**Explanation:** Realized LGD represents the actual loss experienced by the lender when a borrower defaults. It considers the outstanding exposure at the time of default, any recoveries obtained, and the costs associated with the recovery process. Discounting future cashflows to present value is a standard practice in finance.

**Real-world application:** Accurately calculating realized LGD is crucial for financial institutions to estimate potential losses from their loan portfolios, set appropriate capital reserves, and comply with regulatory requirements.

**2.2 Through-The-Cycle (TTC) LGD Model:**

TTC LGD models estimate the long-run average LGD for a given segment, irrespective of current economic conditions. This is often modeled using Beta regression.

The Beta distribution is parameterized by two shape parameters, $\alpha$ and $\beta$, where the mean $\mu$ is given by:

$$\mu = \frac{\alpha}{\alpha + \beta}$$

The Beta regression model links the mean $\mu$ to a set of predictors using a link function $g(.)$, such as the logit link:

$$g(\mu) = X\beta$$

Where:

*   $\mu$ is the mean LGD to be predicted.
*   $X$ is the matrix of predictor variables.
*   $\beta$ are the coefficients to be estimated.
*   $g(.)$ is a link function (logit, probit, etc.) that maps $\mu$ (which is between 0 and 1) to the real number line.

Solving for $\mu$ given the logit link function:

$$\mu = \frac{e^{X\beta}}{1 + e^{X\beta}}$$

**Explanation:** Beta regression is suitable for modeling variables that are bounded between 0 and 1, such as LGD. The link function ensures that the predicted LGD values remain within the valid range. The predictors in the model can include loan characteristics, borrower attributes, and other relevant factors.

**Real-world application:** TTC LGD models provide a stable estimate of long-run average losses, which can be used for stress testing and regulatory capital calculations.

**2.3 Point-In-Time (PIT) LGD Overlay:**

PIT LGD models adjust the TTC LGD estimates to reflect current macroeconomic conditions. This is often achieved using linear regression.

$$LGD_{PIT} = LGD_{TTC} + \beta_1 \cdot Macroeconomic\, Factor_1 + \beta_2 \cdot Macroeconomic\, Factor_2 + ... + \epsilon$$

Where:

*   $LGD_{PIT}$ is the Point-In-Time LGD estimate.
*   $LGD_{TTC}$ is the Through-The-Cycle LGD estimate.
*   $Macroeconomic\, Factor_i$ is the value of the i-th macroeconomic indicator (e.g., unemployment rate, GDP growth).
*   $\beta_i$ is the coefficient for the i-th macroeconomic factor, estimated from historical data.
*   $\epsilon$ is the error term.

**Explanation:** PIT overlays capture the impact of macroeconomic conditions on LGD. By including macroeconomic factors in the model, the LGD estimates can be adjusted to reflect current or forecasted economic conditions.

**Real-world application:** PIT LGD models are used to assess the impact of economic downturns or upturns on potential losses, which can inform risk management decisions and capital planning.

### 3. Code Requirements

**3.1 Libraries:**

*   **pandas:** Data manipulation and analysis. Used for reading, cleaning, and transforming the loan-level data and macro data.
*   **numpy:** Numerical computing. Used for performing mathematical calculations, such as discounting recoveries and calculating LGD.
*   **matplotlib/seaborn:** Data visualization. Used for creating histograms, scatter plots, and other visualizations to explore the data and assess model performance.
*   **scikit-learn:** Machine learning algorithms. Used for building and evaluating the Beta regression model and linear regression models.
*   **statsmodels:** Statistical modeling. May be used for more advanced statistical analysis and model diagnostics.
*   **FRED API (e.g., `pandas_datareader`):** Accessing macroeconomic data from the Federal Reserve Economic Data (FRED) API.
*   **pickle/joblib:** Saving and loading model artifacts.
*   **kaggle:** download datasets

**3.2 Input/Output Expectations:**

*   **Input:** LendingClub loan data (CSV or Parquet format) download it from kaggle from kaggle, macroeconomic data from FRED API. Do not have a load_dataset function.
*   **Output:** Realized LGD values for each loan, TTC LGD model, PIT LGD overlay, model evaluation metrics, visualizations, and saved model artifacts.

**3.3 Algorithms/Functions:**

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

### 4. Additional Notes or Instructions

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

### Functions Required:

### 0) Utilities & Reproducibility

* `set_seed(seed: int = 42) -> None` — fix numpy/sklearn RNG for reproducibility.
* `assert_bounds(x: pd.Series, lo: float = 0.0, hi: float = 1.0) -> None` — raise if any LGD outside \[0,1].
* `validate_required_columns(df: pd.DataFrame, cols: list[str]) -> None` — schema guardrail before compute.

### 1) Data I/O

* `read_lendingclub(path: str, dtypes: dict | None = None) -> pd.DataFrame` — load LC loans.
* `fetch_fred_series(series: dict[str, str], start: str, end: str, api_key: str) -> pd.DataFrame` — UNRATE, GDP, etc.
* `save_parquet(df: pd.DataFrame, path: str) -> None` / `read_parquet(path: str) -> pd.DataFrame` — dataset artifacts.

### 2) Default, EAD & Recovery Cashflows

* `filter_defaults(df: pd.DataFrame, default_statuses: set[str]) -> pd.DataFrame` — keep defaulted loans only.
* `assemble_recovery_cashflows(df: pd.DataFrame) -> pd.DataFrame` — per-loan rows with recovery **amounts & dates** and collection fees.
* `compute_ead(row: pd.Series) -> float` — exposure at default (installment context).
* `pv_cashflows(cf: pd.DataFrame, eff_rate: float, default_date: pd.Timestamp) -> tuple[float, float]` — PV(recoveries), PV(collection costs). 
* `compute_realized_lgd(ead: float, pv_rec: float, pv_cost: float) -> float` — (EAD − PV(rec) − PV(cost)) / EAD.

### 3) Segmentation & Feature Engineering

* `assign_grade_group(grade: str) -> str` — “Prime” (A–B) vs “Subprime” (C–G).
* `derive_cure_status(df: pd.DataFrame) -> pd.Series` — cured vs not cured (and restructured flag if available).
* `build_features(df: pd.DataFrame) -> pd.DataFrame` — numeric drivers (loan amount, int rate, term, time-to-default, etc.).

### 4) Splits & Cohorts

* `add_default_quarter(df: pd.DataFrame, default_date_col: str) -> pd.Series` — quarterly cohort key for PIT mapping.
* `temporal_split(df: pd.DataFrame, train_span: tuple, val_span: tuple, oot_span: tuple) -> tuple[pd.DataFrame,...]` — create OOT set and save it.

### 5) TTC Model (LGD on \[0,1])

* `fit_beta_regression(X: pd.DataFrame, y: pd.Series) -> Any` — Beta regression via `statsmodels` for bounded target.
* `predict_beta(model: Any, X: pd.DataFrame) -> np.ndarray` — TTC LGD predictions (clip to \[0,1]).
* `apply_lgd_floor(lgd: pd.Series, floor: float = 0.05) -> pd.Series` — enforce regulatory floor.

*(If data are sparse, a fallback `fit_fractional_logit(...)` can mirror the TTC step.)*

### 6) PIT Overlay (Macro Adjustment)

* `aggregate_lgd_by_cohort(df: pd.DataFrame) -> pd.DataFrame` — average realized/TTC LGD by quarter.
* `align_macro_with_cohorts(macro: pd.DataFrame, cohorts: pd.DataFrame, lag_q: int = 1) -> pd.DataFrame` — join UNRATE/GDP to quarters.
* `fit_pit_overlay(ttc_avg: pd.Series, macro_df: pd.DataFrame) -> Any` — linear model: LGDₜ = a + b·macro.
* `apply_pit_overlay(ttc_pred: pd.Series, macro_row: pd.Series, coefs: dict, mode: str = "additive") -> pd.Series` — add/multiply overlay per segment.

### 7) Evaluation

* `mae(y_true: np.ndarray, y_pred: np.ndarray) -> float` — headline accuracy metric.
* `calibration_bins(y_true: pd.Series, y_pred: pd.Series, n_bins: int = 10) -> pd.DataFrame` — binned mean predicted vs. actual; plotting helper.
* `residuals_vs_fitted(model: Any, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame` — diagnostics for TTC fit.

### 8) Visualizations (Notebook-Ready)

* `plot_lgd_hist_kde(df: pd.DataFrame, by: str | None = None) -> None` — distribution overall/by group.
* `plot_box_violin(df: pd.DataFrame, x: str, y: str = "LGD_realized") -> None` — term/cure comparisons.
* `plot_corr_heatmap(df: pd.DataFrame, cols: list[str]) -> None` — driver correlations.
* `plot_mean_lgd_by_grade(df: pd.DataFrame) -> None` — justify grade segmentation.
* `plot_pred_vs_actual(y_true: pd.Series, y_pred: pd.Series) -> None` — 45° scatter.
* `plot_calibration_curve(bins_df: pd.DataFrame) -> None` — calibration check.
* `plot_quarterly_lgd_vs_unrate(lgd_q: pd.Series, unrate_q: pd.Series) -> None` — dual-axis trend.

### 9) Artifacts & Exports

* `save_model(obj: Any, path: str) -> None` / `load_model(path: str) -> Any` — pickle/joblib models & preprocessors.
* `export_oot(df_oot: pd.DataFrame, path: str) -> None` — out-of-time holdout.
* `save_quarterly_snapshots(df: pd.DataFrame, dirpath: str) -> None` — `snap_YYYYQ.csv` for PSI, migrations, etc.
* `write_macro_scenarios_json(scenarios: dict, path: str) -> None` — baseline/adverse/severely-adverse inputs.

