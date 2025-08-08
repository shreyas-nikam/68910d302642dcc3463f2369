id: 68910d302642dcc3463f2369_user_guide
summary: Lab 3.1: LGD Models - Development User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Loss Given Default (LGD) Modeling Codelab

This codelab provides a hands-on guide to understanding and building Loss Given Default (LGD) models, a crucial component of credit risk management. LGD represents the expected loss on an exposure in the event of a borrower's default, considering potential recoveries. We'll explore both Through-The-Cycle (TTC) and Point-In-Time (PIT) approaches, learning how to prepare data, build statistical models, and incorporate macroeconomic factors. This application will help you understand these concepts in a visual manner.

## Data Ingestion
Duration: 00:05

This step focuses on loading the LendingClub dataset, which will serve as the foundation for our LGD modeling process. You'll learn how to import data into the application, whether from a local CSV file or by loading a pre-built sample dataset.  The data ingestion step is a pre-requisite for all the subsequent steps.

1.  **Upload Dataset:**  Use the "Upload LendingClub Dataset (CSV)" file uploader to import a CSV file containing LendingClub loan data from your computer.
2.  **Sample Data:** If you don't have a dataset readily available, click the "Load Sample LendingClub Data" button to automatically fetch a sample dataset.  This is a great way to get started and explore the application's features.
3.  **Dataset Overview:** Once the data is loaded (either uploaded or fetched), the application will display the first few rows and the dimensions (number of rows and columns) of the dataset. It will also show all the column names that are present in the dataset, so that you are aware of the variables.
4.  **Dataset Columns Overview**: This is a list of the columns that are present in the data. You can use this to map to your uploaded data and see if any column is missing, or to understand what variables are available for feature engineering in the next step.

<aside class="positive">
<b>Tip:</b>  Ensure that your uploaded CSV file has a structure similar to the LendingClub dataset for optimal compatibility with the subsequent steps.  If you are unsure about the format, you can always start with loading the sample dataset, look at the column names and structure of the data, and then upload your data with the similar format.
</aside>

## Feature Engineering
Duration: 00:15

This step covers the critical process of preparing the raw loan data for LGD modeling.  We'll perform several key transformations and calculations to derive meaningful features.

1.  **Loan Status Selection:** The LGD modeling process requires us to focus on the data which has defaulted. Select relevant loan statuses (e.g., "Charged Off") that represent defaulted loans.  This selection is used to filter the data to include only defaulted loans.
2.  **Discount Rate Input:** The discount rate is used for calculating the present value of cash flows.  Enter a discount rate (e.g., 0.05 for 5%) to be used in the present value calculation of future recoveries. This accounts for the time value of money.
3.  **Feature Engineering Execution:** Click the "Perform Feature Engineering" button to trigger the feature engineering pipeline. The application will then perform the following actions in sequential order.
    *   **Filtering:** Defaulted loans are kept, and non-defaulted loans are filtered out.
    *   **Recovery Cashflow Assembly:** The total and net recoveries are computed by aggregating all the payments that have been made.
    *   **EAD Computation:** Exposure at Default (EAD) is calculated, representing the outstanding balance at the time of default.
    *   **Present Value Calculation:** The present value (PV) of cashflows is calculated using the provided discount rate.
    *   **Realized LGD Computation:** The realized LGD is then calculated using the formula:  $LGD = \frac{EAD - PV_{Recoveries}}{EAD}$.
    *   **Feature Creation:** Finally, additional features like "grade\_group" (Prime vs. Sub-prime) and "cure\_status" (Cured vs. Not Cured) are created to further enhance the modeling.
    *   **Feature Engineering:** Additional variables like `emp_length` are converted to numeric. Also one-hot encoding is done for categorical variables.
4.  **Processed Data Preview:**  After the feature engineering is complete, the application displays the first few rows of the processed data, along with its dimensions.  This allows you to inspect the results of the transformations and verify their correctness. You can also view the summary statistics for Realized LGD column, like mean, median etc.

<aside class="negative">
<b>Warning:</b> Ensure that you select at least one loan status to define as "defaulted." Otherwise, the application will not be able to filter the data correctly.
</aside>

## EDA and Segmentation
Duration: 00:10

This step dives into Exploratory Data Analysis (EDA) and data segmentation. This will help to understand the distribution of LGD and identify potential relationships between LGD and other loan characteristics.

1.  **Distribution of Realized LGD:** Visualize the distribution of the "LGD\_realised" variable using histograms and kernel density plots. You can view the overall distribution or segment it by the "grade\_group" (Prime vs. Sub-prime) to see how the distribution varies across these segments.
2.  **LGD vs. Loan Characteristics:** Explore the relationship between "LGD\_realised" and other categorical features like "term" and "cure\_status" using box and violin plots. This can reveal how LGD varies across different loan terms and cure statuses.
3.  **Correlation Heatmap:** Examine the correlation heatmap of numerical features.  This helps to identify multicollinearity among the features and understand how they are correlated with "LGD\_realised".  Understanding correlations can guide feature selection in the modeling process.
4.  **Mean LGD by Grade:** Analyze the bar chart showing the average "LGD\_realised" for each loan grade.  This illustrates the impact of credit quality (as reflected by the loan grade) on LGD.
5.  **Interactive Data Filtering:** Use the interactive slider to filter the data based on numerical columns and observe the impact on the "LGD\_realised" distribution. This allows you to dynamically explore the data and understand the relationships between variables. Choose a numerical column, and then a filter will appear. You can slide the filter in a range of minimum and maximum value of that variable and observe how the data distribution changes.
6.  **EDA Findings Summary**: You can also look at the summary of EDA findings that tell you about the relationship of data distribution with various variables.

<aside class="positive">
<b>Tip:</b>  Use the correlation heatmap to identify features that are strongly correlated with LGD and might be good predictors for the LGD model.
</aside>

## TTC Model Building
Duration: 00:20

Here, we'll build a Through-The-Cycle (TTC) LGD model. This model aims to estimate LGD that is stable over economic cycles, instead of being sensitive to short term changes.

1.  **Feature Selection:** Choose the features to include in the TTC model.  Carefully select the independent variables that you believe are most relevant for predicting LGD. Choosing more features may not always result in a better model.
2.  **LGD Floor Input:**  Enter an LGD floor value (between 0 and 1). This sets a minimum LGD value to ensure conservative estimates. This parameter is also important, as there could be cases of negative LGD that need to be handled properly.
3.  **Model Training:** Click the "Train TTC LGD Model" button to train the Beta regression model.  The application will fit a Beta regression model to the historical data, using the selected features and the specified LGD floor.
4.  **Model Summary Display:** Once the model is trained, the application will show the model summary. You can use the model summary to look at the variable coefficients and p-values, and see if the model is statistically significant.
5.  **Model Evaluation and Visualization:** Evaluate the model's performance using various plots:
    *   **Predicted vs. Actual:**  A scatter plot of predicted LGD vs. actual LGD values.  The closer the points are to the 45-degree line, the better the model's predictive accuracy.
    *   **Calibration Curve:** A plot showing the relationship between predicted and actual LGD values across different bins.  A well-calibrated model will have a calibration curve close to the 45-degree line.
    *   **Residuals vs. Fitted Values:**  A scatter plot of residuals (the difference between actual and predicted values) against the fitted (predicted) values.  This helps to identify any patterns in the residuals, which could indicate problems with the model.
6.  **Mean Absolute Error**: The model will also output Mean Absolute Error, which tells the average magnitude of error in the predictions of the model.
7.  **Temporal Split**: The dataset will be split into training and OOT (Out of Time) set, and MAE and model evaluation plots are created for both.
8. **Model Export:** Once satisfied with the TTC model, you can download the trained model artifact for future use or deployment.

<aside class="negative">
<b>Warning:</b>  Ensure that you select at least one feature for model training.  The application will not be able to train the model without any input features.
</aside>

## PIT Overlay
Duration: 00:20

In this step, we'll build a Point-In-Time (PIT) overlay model. This complements the TTC model by adjusting LGD estimates based on current macroeconomic conditions.

1.  **Aggregate LGD by Cohort:**  The application aggregates realized LGD by time cohorts (e.g., quarter) to observe the historical trends in LGD.
2.  **Load Macroeconomic Data:**  The application loads macroeconomic data, such as unemployment rate and GDP growth.  These indicators will be used to capture the influence of the economic environment on LGD.
3.  **Align Data:** The aggregated LGD data is merged with the macroeconomic data based on the corresponding time periods, aligning the LGD trends with the economic indicators.
4.  **Train PIT Overlay Model:** An ordinary least squares (OLS) regression model is trained to capture the relationship between LGD deviations from the TTC level and the selected macroeconomic indicators.
5.  **Select Macroeconomic Indicators**: Select the macroeconomic indicators that you wish to add as a variable to the model.
6.  **Analyze Model Output**: Analyze the model output and visualizations that are present.
7.  **Apply Stress Testing**: After the model is built, you can also apply a stress testing scenario, which will show how LGD is affected by changing a macroeconomic factor. A chart showing the trends in LGD is also presented.

<aside class="positive">
<b>Tip:</b>  Experiment with different macroeconomic indicators to see which ones have the strongest impact on LGD deviations.
</aside>

## Model Evaluation
Duration: 00:10

This step focuses on assessing the performance of both the TTC and PIT LGD models.

1.  **Load Data:** The application automatically loads the necessary data, including the processed loan data, TTC model predictions, and PIT model predictions (if available).
2.  **TTC Model Evaluation:** The application calculates the Mean Absolute Error (MAE) for the TTC model and displays the result. It also generates a histogram of the TTC model errors to visualize the error distribution.
3.  **PIT Model Evaluation:** If a PIT model has been trained, the application calculates the MAE for the PIT model and displays the result. It also generates a histogram of the PIT model errors.
4.  **TTC vs. PIT Comparison:** A bar chart is created to compare the MAE of the TTC and PIT models, providing a visual comparison of their performance.
5.  **Model Evaluation Metrics**: The application will output metrics like Mean Absolute Error, and also charts like Distribution of Model Errors.

<aside class="negative">
<b>Warning:</b>  Ensure that both the TTC and PIT models have been trained before attempting to evaluate them.
</aside>

## Model Export
Duration: 00:05

This step allows you to export the trained LGD models (TTC and PIT overlay) and the processed dataset for external use, auditing, or deployment.

1.  **Download Processed Data:** Click the "Download Processed Data as CSV" button to download the processed loan data (after feature engineering) as a CSV file.
2.  **Download TTC Model:** Click the "Download Trained TTC LGD Model" button to download the trained TTC LGD model as a `.joblib` file.
3.  **Download PIT Overlay Model:** Click the "Download Trained PIT LGD Overlay Model" button to download the trained PIT LGD overlay model as a `.joblib` file.

<aside class="positive">
<b>Tip:</b>  The downloaded `.joblib` files can be loaded back into a Python environment using the `joblib.load('filename.joblib')` function.
</aside>
