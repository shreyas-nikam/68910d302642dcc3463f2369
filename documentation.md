id: 68910d302642dcc3463f2369_documentation
summary: Lab 3.1: LGD Models - Development Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Building Loss Given Default (LGD) Models

This codelab guides you through building a Loss Given Default (LGD) model using the QuLab application. LGD represents the expected loss when a borrower defaults on a loan. Accurate LGD models are crucial for credit risk management, portfolio valuation, and regulatory compliance.

In this lab, you will learn to:

*   Ingest and prepare loan data.
*   Engineer features relevant to LGD.
*   Perform exploratory data analysis (EDA) to understand data characteristics.
*   Develop a Through-The-Cycle (TTC) LGD model.
*   Implement a Point-In-Time (PIT) overlay using macroeconomic factors.
*   Evaluate model performance.
*   Export model artifacts.

This application uses Streamlit, Pandas, and Plotly to provide an interactive environment for LGD modeling.  We will cover both theoretical concepts and practical implementation details. The core idea is to build a foundation for understanding LGD, then extend it with more sophisticated techniques like macroeconomic overlays.

**Key Concepts:**

*   **LGD (Loss Given Default):** The percentage of exposure lost if a borrower defaults. Calculated as $LGD = 1 - \text{Recovery Rate}$.
*   **Recovery Rate:**  The percentage of the exposure that is recovered after a default. Calculated as $\frac{\text{Recoveries}}{\text{Exposure at Default (EAD)}}$.
*   **TTC (Through-The-Cycle):** A model that aims to be stable across different economic conditions, typically built using long-term average relationships.
*   **PIT (Point-In-Time):** A model that incorporates current economic conditions to provide a more up-to-date LGD estimate.
*   **EAD (Exposure at Default):** The outstanding balance of a loan at the time of default.

## Data Ingestion
Duration: 00:05

This step involves uploading the loan dataset in CSV format. The application supports loading data directly from your local machine.

1.  Navigate to the "Data Ingestion" page using the sidebar.
2.  Click on "Choose a CSV file" and select your LendingClub dataset.  Make sure the CSV file has column headers.
3.  Once the file is uploaded and successfully parsed, a success message will appear, and the dataframe will be displayed. The data is also stored in the Streamlit session state for use in subsequent steps.

<aside class="positive">
<b>Tip:</b> Ensure your CSV file is properly formatted and contains relevant loan information, including loan status, recovery amounts, and loan characteristics.
</aside>

## Feature Engineering
Duration: 00:10

In this step, you will filter the data based on loan status and specify a discount rate. Feature engineering is essential for creating relevant variables that influence LGD.

1.  Navigate to the "Feature Engineering" page.
2.  Select the desired loan statuses. The default selection is "Charged Off," representing loans that have defaulted.
3.  Enter a discount rate.  This rate is used to calculate the present value of recoveries. The default value is 0.05 (5%). The discount rate reflects the time value of money.
4.  The filtered data will be displayed below.  This filtered dataframe is stored in the session state to be used by subsequent steps.

<aside class="negative">
<b>Warning:</b> Ensure that you select the appropriate loan statuses for your analysis. Selecting incorrect statuses can lead to inaccurate LGD estimates.
</aside>

## EDA and Segmentation
Duration: 00:15

Exploratory Data Analysis (EDA) helps you understand the characteristics of your data and identify potential segments.

1.  Navigate to the "EDA and Segmentation" page.
2.  You'll see a histogram of `loan_amnt` providing an overview of the loan amounts distribution.
3.  Select a numerical feature from the dropdown to visualize its distribution using a box plot.
4.  Use the slider to filter the data based on the selected feature. This allows you to explore different segments of the data.

<aside class="positive">
<b>Tip:</b> Use EDA to identify potential drivers of LGD and to understand the relationships between different variables. Look for segments with significantly different LGD characteristics.
</aside>

## TTC Model Building
Duration: 00:30

This step involves building a Through-The-Cycle (TTC) LGD model. You will perform data preparation, feature selection, model training, and evaluation.

1.  Navigate to the "TTC Model Building" page.
2.  The application automatically performs several data preparation steps, including filtering for defaulted loans, assembling recovery cashflows, computing EAD, calculating the present value of recoveries, and computing realized LGD.
3.  Select the features to use for training the TTC model.  Consider factors like loan amount, interest rate, debt-to-income ratio, and annual income.
4.  Specify the training data split ratio. The default value is 0.7, meaning 70% of the data will be used for training and 30% for testing.
5.  Enter an LGD floor value.  This sets a minimum value for the predicted LGD. The default value is 0.01.
6.  Click the "Train Model" button to train the Beta regression model.

**Data Preparation Steps:**

*   **Filter Defaults:**  Keep only the defaulted loans using the `filter_defaults` function.
*   **Assemble Recovery Cashflows:** Calculates the total recoveries for each loan using the `assemble_recovery_cashflows` function.
*   **Compute EAD:**  Calculates the Exposure at Default (EAD) for each loan using the `compute_ead` function.
*   **PV Cashflows:** Calculates the present value of cashflows using the `pv_cashflows` function.
*   **Compute Realized LGD:**  Calculates the realized Loss Given Default (LGD) for each loan using the `compute_realized_lgd` function. The LGD is calculated as $1 - \frac{\text{Total Recoveries}}{\text{EAD}}$. This value is then clipped between 0 and 1.

**Temporal Split:**
The data is split into training and testing sets based on time using the `temporal_split` function. This ensures that the model is evaluated on data from a later period, simulating real-world deployment.

**Beta Regression:**
A Beta regression model is used to predict LGD values.  Since LGD is a rate between 0 and 1, Beta regression is appropriate.  Since there is no readily available Beta Regression in scikit-learn, we use the `LinearRegression` model from `scikit-learn` as a proxy.

**Model Evaluation:**
After training, the model's performance is displayed using plots of predicted vs. actual LGD values, a calibration curve, and a residuals vs. fitted plot.

<aside class="negative">
<b>Warning:</b>  The quality of the TTC model depends on the features selected and the data used for training.  Carefully consider the relevance of each feature and ensure that the data is representative of the portfolio you are modeling.
</aside>

## PIT Overlay
Duration: 00:25

This step involves incorporating macroeconomic factors to create a Point-In-Time (PIT) overlay for the TTC LGD model.  This adjusts the TTC LGD based on current economic conditions.

1.  Navigate to the "PIT Overlay" page.
2.  The application aggregates LGD by cohort (e.g., loan origination quarter).
3.  Import macroeconomic data. In this example, sample macroeconomic data with unemployment rate is used.
4.  The application aligns the macroeconomic data with the LGD cohorts based on time.
5.  Visualize the relationship between LGD and the unemployment rate.
6.  Use the slider to simulate a stress scenario with a higher unemployment rate.  The application calculates the stressed LGD based on the PIT overlay model.

**Process:**

1.  **Aggregate LGD by Cohort:** The `aggregate_lgd_by_cohort` function groups the data by loan origination quarter and calculates the average LGD for each cohort.
2.  **Align Macro Data with LGD Cohorts:**  The `align_macro_with_cohorts` function merges the LGD cohort data with the macroeconomic data based on the quarter.
3.  **PIT Overlay Model:** A PIT overlay model (again using `LinearRegression` as a proxy) is trained to adjust the TTC LGD based on the unemployment rate.
4.  **Scenario Analysis:** Allows you to explore the impact of different unemployment rate scenarios on the LGD.

<aside class="positive">
<b>Tip:</b>  Experiment with different macroeconomic factors and explore their impact on LGD. Consider factors like GDP growth, inflation, and interest rates.
</aside>

## Model Evaluation
Duration: 00:10

This step provides a more detailed evaluation of the TTC LGD model.

1.  Navigate to the "Model Evaluation" page.
2.  The application calculates the Mean Absolute Error (MAE) on the test set.
3.  The predicted vs actual LGD, Calibration Curve, and Residuals vs Fitted Plot are displayed to help you assess the model's performance.

<aside class="positive">
    <b>Info:</b> Focus on the MAE, the closer the predictions are to the actuals better the model. Also analyze the different plots generated to look for bias, overestimation, or underestimation in the model.
</aside>

## Model Export
Duration: 00:05

This step allows you to download the trained TTC model and the test dataset.

1.  Navigate to the "Model Export" page.
2.  Click the "Download Test Dataset (CSV)" link to download the test dataset.  Downloading the model is displayed as a placeholder.

<aside class="negative">
<b>Warning:</b> The model export functionality is a placeholder. In a real-world scenario, you would need to implement proper serialization and deserialization of the model object.
</aside>

This concludes the QuLab codelab. You have learned how to build a Loss Given Default (LGD) model using the QuLab application, covering data ingestion, feature engineering, EDA, TTC model building, PIT overlay, model evaluation, and model export.
