id: 68910d302642dcc3463f2369_documentation
summary: Lab 3.1: LGD Models - Development Documentation
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: A Comprehensive Guide to Loss Given Default (LGD) Modeling

This codelab provides a comprehensive guide to building and evaluating Loss Given Default (LGD) models using a Streamlit application called QuLab.  LGD represents the expected loss if a borrower defaults on a loan. Accurate LGD models are crucial for effective credit risk management, regulatory compliance, and informed decision-making in lending institutions.

**Importance:**  LGD models directly impact risk-weighted assets and capital adequacy calculations.  Improving LGD model accuracy can lead to significant capital savings and more efficient resource allocation. This application enables users to explore various aspects of LGD modeling, including data preparation, feature engineering, model building (TTC and PIT), and thorough model evaluation.

**Concepts Explained:** This codelab covers the following key concepts:

*   **Loss Given Default (LGD):**  The percentage of loss an organization incurs if a borrower defaults.
*   **Through-The-Cycle (TTC) Models:** LGD models that are stable and less sensitive to short-term economic fluctuations.
*   **Point-In-Time (PIT) Models:** LGD models that incorporate current economic conditions and are more dynamic.
*   **Exposure at Default (EAD):** The outstanding balance of a loan at the time of default.
*   **Data Ingestion, Feature Engineering, and EDA:** Necessary steps to prepare data for building robust LGD models.
*   **Model Evaluation Metrics:** Techniques to assess the performance and reliability of LGD models.
*   **Beta Regression:** A statistical modeling technique suitable for predicting values bounded between 0 and 1, such as LGD.

## Data Ingestion
Duration: 00:05:00

This step guides you through the process of loading the LendingClub dataset, which serves as the foundation for developing LGD models.

1.  **Access the Data Ingestion Page:** Navigate to the "Data Ingestion" option in the sidebar navigation menu.

2.  **Dataset Loading:**  You have two options for loading the dataset:

    *   **Upload a CSV file:**  Use the file uploader to select a CSV file containing LendingClub loan data from your local machine.
    *   **Load Default Dataset:** If you don't have a local file, you can load a default LendingClub dataset (2018Q4) directly from a specified URL by clicking the "Load Default LendingClub Data" button.

3.  **Data Preview:**  After successful loading, a preview of the dataset (the first few rows) will be displayed, along with the total number of rows and columns.

<aside class="positive">
The application uses Streamlit's session state (`st.session_state`) to store the loaded DataFrame.  This ensures that the data persists as you navigate between different sections of the application.
</aside>

<aside class="negative">
Ensure that the CSV file is properly formatted and contains the necessary columns for LGD modeling. The default dataset is pre-formatted for compatibility, but custom datasets might require adjustments.
</aside>

## Feature Engineering
Duration: 00:15:00

This step focuses on preprocessing the raw LendingClub data to derive meaningful features required for LGD model development.

1.  **Access the Feature Engineering Page:** Navigate to the "Feature Engineering" option in the sidebar navigation menu.

2.  **Filter Defaulted Loans:**

    *   The application allows you to filter the dataset to include only loans that have defaulted. The default status is 'Charged Off'.
    *   Use the "Select Loan Status to Filter By" dropdown menu to choose the appropriate loan status for LGD calculation.
    *   Click the "Filter" button to apply the filter.

3.  **Assemble Recovery Cashflows and Compute EAD:**

    *   Click the "Assemble Recoveries and Compute EAD" button to execute this step.
    *   This function calculates the Exposure at Default (EAD) for each loan. The simplified EAD is calculated as `loan_amnt - total_rec_prncp`.

4.  **Calculate Realized LGD:**

    *   Click the "Compute Realized LGD" button to calculate the realized Loss Given Default (LGD) for each loan.
    *   The realized LGD is calculated as `(EAD - recovery_amount - collection_costs) / EAD`.  Values are clipped between 0 and 1.

5.  **Assign Grade Group:**

    *   Click the "Assign Grade Group" button to categorize loans into 'Prime' or 'Sub-prime' based on their grade.

6.  **Derive Cure Status:**

    *   Click the "Derive Cure Status" button to determine if a loan has been "cured" (i.e., brought back to a performing status after a period of distress).

7.  **Build Custom Features:**

    *   Select the features you want to build from the multiselect box (e.g., `loan_size_income_ratio`, `int_rate_squared`).
    *   Click the "Build Selected Features" button.
    *   The application will compute these features and add them to the DataFrame.

8.  **Add Default Quarter:**

    *   Click the "Add Default Quarter" button to derive the quarter in which the loan defaulted, based on the issue date.

9.  **Processed Data Preview:**  A preview of the processed DataFrame will be displayed, allowing you to verify the results of the feature engineering steps.

<aside class="positive">
Feature engineering is critical for building accurate and robust LGD models. Experiment with different features to identify the most predictive variables.
</aside>

<aside class="negative">
Ensure that the necessary columns (`recovery_amount`, `collection_recovery_fee`, `loan_amnt`, `total_rec_prncp`, `grade`, `issue_d`) are present in the DataFrame before executing each feature engineering step. Missing columns will lead to errors or incorrect calculations.
</aside>

## EDA and Segmentation
Duration: 00:20:00

This step explores the processed data using Exploratory Data Analysis (EDA) techniques and segments the data for model building and validation.

1.  **Access the EDA and Segmentation Page:** Navigate to the "EDA and Segmentation" option in the sidebar navigation menu.

2.  **Temporal Data Split:**

    *   Split the data into training and Out-of-Time (OOT) samples using the "Select Training Data Proportion" slider.  The OOT sample is crucial for robust model validation.
    *   Click the "Perform Temporal Split" button.

3.  **LGD Realized Distribution:**

    *   Visualize the distribution of `LGD_realized` using histograms and Kernel Density Estimates (KDEs).
    *   Optionally group the distribution by a categorical variable using the "Group LGD Distribution by" dropdown.
    *   Click the "Plot LGD Distribution" button.

4.  **LGD by Categorical Variables:**

    *   Explore `LGD_realized` across different categories using box or violin plots.
    *   Select a categorical column to plot using the "Select Categorical Column" dropdown.
    *   Choose the plot type (box or violin) using the radio buttons.
    *   Click the "Plot LGD by Category" button.

5.  **Correlation Heatmap of Numeric Drivers:**

    *   Understand the relationships between numeric features using a correlation heatmap.
    *   Select the numeric columns to include in the heatmap using the "Select Numeric Columns for Heatmap" multiselect.
    *   Click the "Plot Correlation Heatmap" button.

<aside class="positive">
EDA helps identify patterns, outliers, and relationships in the data, which informs feature selection and model building.  Pay close attention to the distribution of `LGD_realized` and its relationship with other variables.
</aside>

<aside class="negative">
Ensure that the `LGD_realized` column has been calculated in the Feature Engineering stage before attempting to plot its distribution.
</aside>

## TTC Model Building
Duration: 00:25:00

This step guides you through the process of building a Through-The-Cycle (TTC) LGD model using Beta Regression.

1.  **Access the TTC Model Building Page:** Navigate to the "TTC Model Building" option in the sidebar navigation menu.

2.  **Select Features for TTC Model:**

    *   Choose the independent variables (features) to train your LGD model using the multiselect box.
    *   Consider including features like `loan_amnt`, `int_rate`, `dti`, `annual_inc`, `revol_util`, etc.
    *   Set the LGD Floor Value: Use the number input to specify a minimum predicted LGD value. This ensures conservative estimates.

3.  **Train TTC LGD Model:**

    *   Click the "Train Beta Regression Model" button to train the model.
    *   The application will fit a Beta regression model to the training data.

4.  **Model Visualizations (Training Data):**

    *   The application generates several plots to visualize the model's performance on the training data:
        *   **Predicted vs. Actual LGD:**  A scatter plot showing the relationship between predicted and actual LGD values.
        *   **Calibration Curve:**  A plot showing the calibration of the model, i.e., how well the predicted LGD values align with the actual LGD values within different bins.
        *   **Residuals vs. Fitted Values:**  A plot used to assess the model's assumptions about the residuals.

<aside class="positive">
Beta Regression is a suitable choice for modeling LGD because it naturally handles values bounded between 0 and 1.
</aside>

<aside class="negative">
Ensure that you select at least one feature before training the model. The `LGD_realized` column must be present in the training data.  Handle missing values appropriately before training the model.
</aside>

## PIT Overlay
Duration: 00:10:00

This section would ideally overlay Point-In-Time (PIT) adjustments onto the TTC model. However, the current code does not contain the implementation for this functionality. This section would typically involve:

1.  **Loading Macroeconomic Data:**  Ingesting relevant macroeconomic indicators (e.g., GDP growth, unemployment rate) that reflect the current economic conditions.

2.  **Modeling the Relationship:**  Establishing a statistical relationship between the macroeconomic indicators and the LGD. This might involve regression analysis or other modeling techniques.

3.  **Adjusting TTC Predictions:**  Using the modeled relationship to adjust the TTC LGD predictions based on the current values of the macroeconomic indicators.

Since there is no implementation in the provided code, this step is skipped.

## Model Evaluation
Duration: 00:15:00

This step would involve evaluating the performance of the LGD model using appropriate metrics on both the training and OOT datasets.

1.  **Evaluation Metrics:** Common metrics for evaluating LGD models include:

    *   **Mean Absolute Error (MAE):**  The average absolute difference between predicted and actual LGD values.
    *   **Root Mean Squared Error (RMSE):**  The square root of the average squared difference between predicted and actual LGD values.
    *   **R-squared:** A measure of how well the model fits the data.
    *   **Calibration:**  Assessing how well the predicted LGD values align with the actual LGD values across different risk segments.

2.  **OOT Performance:**  It is crucial to evaluate the model's performance on the OOT dataset to assess its generalization ability and robustness to unseen data.

Since there is no implementation in the provided code, this step cannot be completed.  The application would ideally calculate these metrics and display them to the user, along with visualizations to support the evaluation.

## Model Export
Duration: 00:05:00

This step focuses on exporting the trained LGD model for deployment and use in other applications.

1.  **Model Serialization:**  The trained Beta regression model (including its coefficients and other relevant parameters) needs to be serialized into a file format that can be easily stored and loaded. Common serialization formats include:

    *   **Pickle:**  A Python-specific format.
    *   **PMML (Predictive Model Markup Language):**  A standard XML-based format for representing predictive models.

2.  **Export Options:** The application should provide options for exporting the model in different formats.

Since there is no implementation in the provided code, this step cannot be completed. The application would ideally allow the user to download the serialized model file.

<aside class="positive">
Exporting the model allows you to integrate it into your credit risk management systems and use it for making predictions on new loan applications.
</aside>

This codelab provides a foundational understanding of LGD modeling using the QuLab Streamlit application. While some functionalities are missing from the provided code, the guide highlights the key steps and concepts involved in building, evaluating, and deploying LGD models.  By implementing the missing functionalities, the QuLab application can become a powerful tool for credit risk professionals.
