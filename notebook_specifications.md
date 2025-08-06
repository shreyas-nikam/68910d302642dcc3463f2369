
# Through-the-Cycle (TTC) LGD Modeler: Jupyter Notebook Specification

## 1. Notebook Overview

**Learning Goals:**

*   Understand the principles and implementation of Through-the-Cycle (TTC) Loss Given Default (LGD) modeling.
*   Learn how to segment loan portfolios for LGD modeling based on key characteristics.
*   Gain practical experience in building and evaluating TTC LGD models using regression techniques, specifically Beta regression.
*   Understand the importance of regulatory floors in LGD modeling and how to implement them.

**Expected Outcomes:**

*   A Jupyter Notebook that demonstrates the construction and evaluation of a TTC LGD model.
*   The ability to segment a loan portfolio based on defined criteria.
*   A working Beta regression model for predicting LGD within each segment.
*   Implementation of a regulatory LGD floor.
*   Model diagnostic outputs, including pseudo-R² and calibration plots.

## 2. Mathematical and Theoretical Foundations

### 2.1 Loss Given Default (LGD)

LGD represents the proportion of exposure lost on a loan when a borrower defaults. It is calculated as:

$$LGD = \frac{Loss\ Amount}{Exposure\ at\ Default (EAD)}$$

Where:

*   $Loss\ Amount = EAD - Recoveries$
*   $EAD$ is the outstanding balance at the time of default.
*   $Recoveries$ are the amounts recovered from the defaulted loan.

### 2.2 Beta Regression

Beta regression is used to model variables that are bounded between 0 and 1, such as LGD.  The Beta distribution is defined by two parameters, $\alpha$ and $\beta$, which influence the shape of the distribution. The mean ($\mu$) and precision ($\phi$) are related to these parameters by:

$$\mu = \frac{\alpha}{\alpha + \beta}$$

$$\phi = \alpha + \beta$$

The Beta regression model links the mean $\mu$ to a set of predictors using a link function $g(.)$. A common choice is the logit link function:

$$g(\mu) = log(\frac{\mu}{1 - \mu}) = X\beta$$

Where:

*   $X$ is the matrix of predictor variables.
*   $\beta$ is the vector of regression coefficients.

The probability density function (pdf) of the Beta distribution is given by:

$$f(y; \alpha, \beta) = \frac{y^{\alpha-1}(1-y)^{\beta-1}}{B(\alpha, \beta)}$$

Where $B(\alpha, \beta)$ is the Beta function.

### 2.3 Pseudo-R²

Pseudo-R² measures the goodness-of-fit for models where the outcome variable is not normally distributed, such as in logistic or Beta regression. Several versions exist; a common one is McFadden's Pseudo-R², calculated as:

$$Pseudo-R^2 = 1 - \frac{log\ likelihood\ of\ the\ fitted\ model}{log\ likelihood\ of\ the\ null\ model}$$

A higher Pseudo-R² indicates a better fit.

### 2.4 Regulatory Floor

Regulatory guidelines often require a minimum LGD value to be applied to all exposures. This floor is implemented to ensure that models do not underestimate potential losses. In this case, a 5% floor will be applied, meaning any predicted LGD value below 5% will be set to 5%.

### 2.5 Calibration

Calibration assesses how well the model's predictions align with the actual outcomes.  A well-calibrated model's predictions should, on average, match the observed default rates. Calibration plots are used to visualize this, comparing the mean predicted LGD to the mean observed LGD within different risk segments.

## 3. Code Requirements

### 3.1 Expected Libraries

*   **pandas:** For data manipulation and analysis.
*   **numpy:** For numerical computations.
*   **statsmodels:** For statistical modeling, including Beta regression.
*   **matplotlib/seaborn:** For data visualization.
*   **sklearn:** For model evaluation metrics and data splitting.

### 3.2 Input/Output Expectations

*   **Input:**
    *   *LendingClub Loan Statistics* (Kaggle). Download with `kaggle datasets download -d sgpjesus/lending-club-2007-2018`. 
    *     Or download it from https://www.openintro.org/data/csv/loans_full_schema.csv using wget.
*   **Output:**
    *   A Pandas DataFrame with predicted LGD values for each loan.
    *   Model fit statistics (e.g., pseudo-R²).
    *   Calibration plot.
    *   Visualizations of LGD distributions and segmentations.
    *   Saved model artefacts (preprocessor, regression model).

### 3.3 Algorithms/Functions to be Implemented

1.  **Data Loading and Preprocessing:** Function to load the data and perform basic cleaning, including handling missing values and data type conversions.
2.  **LGD Calculation:** Function to calculate realized LGD based on the formula provided in section 2.1. This function should incorporate EAD, recoveries, and handle edge cases like LGD values outside the 0-1 range (clip to 0 and 1).
3.  **Segmentation:** Function to divide the loan portfolio into segments based on user-defined criteria (e.g., `grade_group` and `cure_status`).
4.  **Beta Regression Model Training:** Function to train a Beta regression model on each segment using the `statsmodels` library.  The function should take the segment data and predictor variables as input.
5.  **LGD Prediction:** Function to predict LGD values for a given dataset using the trained Beta regression model.
6.  **Regulatory Floor Implementation:** Function to apply a 5% floor to the predicted LGD values.
7.  **Model Evaluation:** Functions to calculate pseudo-R² and generate calibration plots.  A function to calculate Mean Absolute Error (MAE)
8.  **Visualization Functions:** Functions to create histograms, box plots, scatter plots, and calibration plots to visualize LGD distributions, segmentations, and model performance.

### 3.4 Visualizations

*   **Histogram & Kernel Density:** of `LGD_realised` (overall & by `grade_group`).
*   **Box/Violin Plots:** of LGD vs term, cure status.
*   **Heatmap:** of Pearson/rank correlations among numeric drivers.
*   **Bar chart:** Mean LGD by grade.
*   **Violin Plot:** for cured vs non-cured LGDs.
*   **Predicted-vs-Actual Scatter Plot:** with 45° line.
*   **Calibration Curve:** Binned mean predicted vs. actual.
*   **Residuals vs Fitted Plot:** For Beta model.

## 4. Additional Notes or Instructions

*   **Data Source:** The notebook will use a synthetic LendingClub loan dataset, emulating the structure of the Kaggle dataset.
*   **Segmentation:** Users should be able to easily modify the segmentation criteria to explore different segmentations.
*   **Model Evaluation:** Focus on interpreting the model fit statistics and calibration plot to assess the model's performance and identify potential issues.
*   **Reproducibility:** The notebook should set a random seed (`RANDOM_STATE = 42`) for reproducibility.
*   **Model Artifacts:** Ensure saving the model as per the document in `/models/lgd_preprocess_v1.pkl`, `lgd_beta_regression_v1.pkl`.
*   **Regulatory Compliance:**  Emphasize the importance of the 5% LGD floor and its role in complying with regulatory requirements.
*   **Assumptions:** Assumes a pre-existing dataset with the specified features and format.  Assumes basic familiarity with Python, Pandas, and statistical modeling.
*   **Constraints:**  The model should be limited to the available features in the dataset.
*   **Customization:**  Users should be able to easily change the predictor variables used in the Beta regression model and the segmentation criteria.

