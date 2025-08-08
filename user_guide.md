id: 68910d302642dcc3463f2369_user_guide
summary: Lab 3.1: LGD Models - Development User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# QuLab: Loss Given Default (LGD) Modeling Codelab

This codelab provides a hands-on guide to understanding and building Loss Given Default (LGD) models. LGD is a crucial parameter in credit risk modeling, representing the expected loss if a borrower defaults on a loan. Understanding and accurately predicting LGD is vital for financial institutions to manage risk effectively and comply with regulatory requirements. We will walk through the entire process, from data ingestion and feature engineering to model building, evaluation, and deployment. This application will cover both Through-The-Cycle (TTC) and Point-In-Time (PIT) approaches to LGD modeling.

**Key Concepts:**

*   **Loss Given Default (LGD):** The proportion of exposure lost if a borrower defaults. Calculated as $LGD = 1 - \text{Recovery Rate}$.
*   **Recovery Rate:** The percentage of the outstanding exposure that the lender recovers after a default. Calculated as $\text{Recovery Rate} = \frac{\text{Recoveries}}{\text{Exposure at Default (EAD)}}$.
*   **Through-The-Cycle (TTC) LGD:** A stable, long-term estimate of LGD that is less sensitive to current economic conditions.
*   **Point-In-Time (PIT) LGD:** A dynamic estimate of LGD that reflects current economic conditions.
*   **Exposure at Default (EAD):** The outstanding balance of a loan at the time of default.

## Data Ingestion
Duration: 00:05

In this step, you will upload the LendingClub dataset, which serves as the foundation for our LGD modeling exercise. The application accepts CSV files, and upon successful upload, it displays the data in a dataframe.

1.  Navigate to the "Data Ingestion" page using the sidebar.
2.  Click on the "Choose a CSV file" button and select your LendingClub dataset.
3.  The application will load the data and display it as a dataframe. A success message will appear once loading is complete.
<aside class="positive">
Make sure that the data you upload contains relevant loan information, including loan status, recovery amounts, and loan characteristics.
</aside>

## Feature Engineering
Duration: 00:10

Feature engineering involves transforming raw data into features that can be used in a model. In this step, you'll filter the dataset and select a discount rate.

1.  Navigate to the "Feature Engineering" page.
2.  **Loan Status Filter:** Select the loan statuses you want to include in your analysis. By default, "Charged Off" is selected.
3.  **Discount Rate Input:** Enter a discount rate to be used in present value calculations. The default value is 0.05.
4.  The application will display the filtered data.
<aside class="negative">
Ensure you select the appropriate loan statuses relevant to defaults and recoveries for accurate LGD modeling. An incorrect loan status might skew the model.
</aside>

## EDA and Segmentation
Duration: 00:15

Exploratory Data Analysis (EDA) and segmentation are crucial for understanding the data and identifying patterns. In this step, you'll explore the data through visualizations and segment it based on features.

1.  Navigate to the "EDA and Segmentation" page.
2.  The application displays a histogram of `loan_amnt`.
3.  **Feature Selection for Visualization:** Select a numerical feature from the dropdown menu.
4.  **Slider for filtering:** Use the slider to filter the data based on the selected feature. This allows you to focus on specific segments of the data.
5.  A box plot of the selected feature will be displayed, providing insights into its distribution.
<aside class="positive">
Experiment with different features and filter ranges to uncover hidden patterns and potential segments within the data. This step is important for feature selection and better model building in the next steps.
</aside>

## TTC Model Building
Duration: 00:20

This step focuses on building a Through-The-Cycle (TTC) LGD model. The TTC model provides a stable, long-term estimate of LGD that is less sensitive to current economic conditions.

1.  Navigate to the "TTC Model Building" page.
2.  The application performs several data preparation steps, including filtering defaulted loans, assembling recovery cashflows, and computing EAD.
3.  **Feature Selection for TTC Model:** Select the features you want to use for training the TTC model.
4.  **Training data split (proportion):** Choose what percentage of the data you want to use for training.
5.  **LGD Floor Value:** Enter a minimum LGD value. This is used to apply a floor to the predicted LGD values.
6.  Click the "Train Model" button to train the Beta regression model.
7.  The application will display the model performance on the test set, including a scatter plot of predicted vs. actual LGD and a calibration curve.
<aside class="negative">
The choice of features greatly impacts model performance. Select features that are highly correlated with LGD based on domain knowledge and EDA.
</aside>

## PIT Overlay
Duration: 00:15

This step incorporates macroeconomic factors to adjust the TTC LGD model and create a Point-In-Time (PIT) LGD model. The PIT model reflects current economic conditions, making it more dynamic.

1.  Navigate to the "PIT Overlay" page.
2.  The application aggregates LGD by cohort and aligns it with macroeconomic data (unemployment rate).
3.  The aligned data is visualized using a dual-axis plot, showing LGD and unemployment rate by quarter.
4.  **Unemployment Rate (Stress):** Use the slider to simulate different unemployment rate scenarios.
5.  The application calculates and displays the stressed LGD based on the selected unemployment rate.
<aside class="positive">
Consider different macroeconomic factors, such as GDP growth, inflation, and interest rates, to enhance the PIT overlay model.
</aside>

## Model Evaluation
Duration: 00:10

This step evaluates the performance of the built TTC model.

1.  Navigate to the "Model Evaluation" page.
2.  The application displays the Mean Absolute Error (MAE) as an evaluation metric.
3.  The application displays the model performance on the test set, including a scatter plot of predicted vs. actual LGD, a calibration curve, and Residuals vs. Fitted Plot.
<aside class="negative">
It is important to consider multiple evaluation metrics and visualizations to get a comprehensive understanding of model performance.
</aside>

## Model Export
Duration: 00:05

This step allows you to export the trained model and test dataset for further analysis or deployment.

1.  Navigate to the "Model Export" page.
2.  You can download the test dataset as a CSV file. Functionality to download the model is a placeholder and will be implemented in the future.
<aside class="positive">
Make sure to document the model and dataset properly, including feature definitions, model parameters, and performance metrics.
</aside>
