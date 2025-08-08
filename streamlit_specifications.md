
# Streamlit Application Requirements Specification

## 1. Application Overview

This Streamlit application aims to provide an interactive tool for exploring and visualizing Loss Given Default (LGD) models. The learning goals are:

- Understand the components of Loss Given Default (LGD) models, including Through-The-Cycle (TTC) and Point-In-Time (PIT) approaches.
- Develop skills in data extraction, cleaning, and feature engineering for credit risk modeling.
- Apply statistical modeling techniques to estimate LGD.
- Learn how to incorporate macroeconomic factors into LGD models.
- Evaluate model performance and prepare artifacts for deployment.

## 2. User Interface Requirements

### Layout and Navigation Structure

The application will consist of a multi-page layout, with sections corresponding to the different steps of LGD model development:
- **Data Ingestion:** Load and display the LendingClub dataset.
- **Feature Engineering:** Filter, transform, and create relevant features.
- **EDA and Segmentation:** Explore the data and identify potential segments.
- **TTC Model Building:** Build a Through-The-Cycle (TTC) LGD model.
- **PIT Overlay:** Incorporate macroeconomic factors for Point-In-Time (PIT) adjustments.
- **Model Evaluation:** Evaluate model performance and visualize results.
- **Model Export:** Download saved artifacts.

Each section will be a separate page, accessible via a sidebar navigation menu.

### Input Widgets and Controls

- **Data Ingestion:**
  - File uploader for uploading the LendingClub dataset (CSV format).
- **Feature Engineering:**
  - Selection boxes to choose loan status for filtering (default: Charged Off).
  - Numerical input fields for specifying discount rates.
- **EDA and Segmentation:**
  - Selection boxes for choosing features to visualize (histograms, box plots, violin plots).
  - Sliders for filtering data based on numerical columns.
- **TTC Model Building:**
  - Selection boxes to choose features for training the TTC model.
  - Numerical input fields for the LGD floor value.
- **PIT Overlay:**
  - Selection boxes to choose macroeconomic indicators.
  - Numerical input fields for stress testing scenarios.
- **Model Evaluation:**
  - Selection boxes to choose evaluation metrics.
- **Model Export:**
  - Buttons to download saved model artifacts and datasets.

### Visualization Components

- **Data Ingestion:**
  - Dataframe display of the loaded dataset.
- **Feature Engineering:**
  - Dataframe display of the filtered and transformed data.
- **EDA and Segmentation:**
  - Histograms and kernel density plots of `LGD_realised` (overall & by `grade_group`).
  - Box/violin plots of LGD vs. term, cure status.
  - Heat-map of pearson / rank correlations among numeric drivers.
  - Bar chart: mean LGD by grade.
  - Violin plot for cured vs. non-cured LGDs.
- **TTC Model Building:**
  - Predicted-vs-actual scatter with 45° line.
  - Calibration curve (binned mean predicted vs. actual).
  - Residuals vs. fitted for Beta model.
- **PIT Overlay:**
  - Dual-axis line chart: quarterly average LGD and unemployment rate.
  - Scenario slider to visualise stressed LGD uplift.
- **Model Evaluation:**
  - Numerical display of evaluation metrics (MAE).

### Interactive Elements and Feedback Mechanisms

- Progress bars during data loading and processing.
- Success/error messages for each step.
- Tooltips for interactive charts and graphs.
- Download links for model artifacts and processed datasets.
- Information boxes explaining the purpose of each section and widget.

## 3. Additional Requirements

### Annotation and Tooltip Specifications

- Charts should include titles, axis labels, and legends.
- Hovering over data points in charts should display relevant information (e.g., exact values).
- Input widgets should have tooltips explaining their purpose and expected input format.

### State Persistence

The Streamlit application should properly save the states of the fields so that changes are not lost when navigating between pages or refreshing the application. This includes:
- Input widget values
- Dataframes that have been processed.
- Model selections.

## 4. Notebook Content and Code Requirements

The Streamlit application will incorporate the following code and functionality from the provided Jupyter Notebook:

### Data Ingestion

- **`set_seed(seed)`:** Sets the random seed for reproducibility.
  ```python
  import random

  def set_seed(seed):
      """Sets the random seed for reproducibility."""
      if not isinstance(seed, int):
          raise TypeError("Seed must be an integer.")
      random.seed(seed)
  ```
  This function should be called at the beginning of the application with a fixed seed (e.g., 42).

- **`fetch_lendingclub_date()`:** Fetches LendingClub loan data from a specified URL.
  ```python
  import pandas as pd
  import requests
  import zipfile
  import io

  def fetch_lendingclub_date():
      """Fetches LendingClub loan data."""
      url = "https://resources.lendingclub.com/LoanStats_2018Q4.csv.zip"  # Example URL, might need updating
      try:
          response = requests.get(url)
          response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
          zip_content = response.content
      except requests.exceptions.RequestException as e:
          raise Exception(f"Connection error: {e}")

      try:
          with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_file:
              csv_files = zip_file.namelist()
              if not csv_files:
                  raise Exception("No CSV file found in the zip archive.")
              csv_file_name = csv_files[0]  # Assuming only one CSV file
              with zip_file.open(csv_file_name) as csv_file:
                  df = pd.read_csv(csv_file, skiprows=1)
                  # Drop the last row if it's completely empty (summary row)
                  df = df.dropna(how='all')
                  return df
      except pd.errors.ParserError as e:
          raise pd.errors.ParserError(f"CSV parsing error: {e}")
      except Exception as e:
          raise Exception(f"Error processing zip file: {e}")
  ```
  Implement a file uploader widget so that the user can upload the dataset from different local paths.

### Feature Engineering

- **`filter_defaults(df)`:** Filters the dataset to include only defaulted loans.
  ```python
  import pandas as pd

  def filter_defaults(df):
      """Filters the dataset to include only defaulted loans."""

      if not isinstance(df, pd.DataFrame):
          raise TypeError("Input must be a Pandas DataFrame.")

      defaulted_loans = df[df['loan_status'] == 'Charged Off']
      return defaulted_loans
  ```
  Implement a selection box to choose the `loan_status` value to filter by.

- **`assemble_recovery_cashflows(df)`:** Assembles recovery cashflows from loan data.
  ```python
  import pandas as pd

  def assemble_recovery_cashflows(df):
      """Assembles recovery cashflows from loan data."""

      if df.empty:
          return pd.DataFrame()

      # Check if required columns exist
      required_columns = ['loan_id', 'recovery_amount', 'collection_costs', 'outstanding_principal']
      for col in required_columns:
          if col not in df.columns:
              raise KeyError(f"Column '{col}' missing in DataFrame.")

      recovery_cashflows = df[['loan_id', 'recovery_amount', 'collection_costs']].copy()

      return recovery_cashflows
  ```
  Error handling for missing columns must be implemented properly.

- **`compute_ead(df)`:** Computes the Exposure at Default (EAD) for each loan.
  ```python
  import pandas as pd

  def compute_ead(df):
      """Computes the Exposure at Default (EAD) for each loan.
      Args:
          df: Pandas DataFrame containing the loan data.
      Output: Pandas DataFrame with EAD calculated.
      """

      if not isinstance(df, pd.DataFrame):
          raise AttributeError("Input must be a Pandas DataFrame.")

      return df
  ```
  This will extract or compute the EAD from the loan data.

- **`pv_cashflows(df, discount_rate)`:** Calculates the present value of cashflows.
  ```python
  import pandas as pd

  def pv_cashflows(df, discount_rate):
      """Calculates the present value of cashflows.
      Args:
          df: Pandas DataFrame containing the cashflow and time data.
          discount_rate: Discount rate for present value calculation.
      Returns:
          Pandas DataFrame with an additional 'present_value' column.
      """
      if not isinstance(discount_rate, (int, float)):
          raise TypeError("Discount rate must be a numeric value.")

      df['present_value'] = df.apply(lambda row: row['cashflow'] / (1 + discount_rate)**row['time'], axis=1)
      return df
  ```
  The discount rate should be taken from an input widget.
  Latex display of the equation will be required:
  $$PV = \frac{CF}{(1 + r)^t}$$
  Where:
  *   $PV$ is the present value of the cashflow
  *   $CF$ is the future cashflow amount
  *   $r$ is the discount rate
  *   $t$ is the time period (in years) until the cashflow is received

- **`compute_realized_lgd(df)`:** Computes the realized Loss Given Default (LGD) for each loan.
  ```python
  import pandas as pd

  def compute_realized_lgd(df):
      """Computes the realized Loss Given Default (LGD) for each loan.

      Args:
          df: Pandas DataFrame containing the loan data with EAD, recoveries and collection costs.

      Returns:
          Pandas DataFrame with realized LGD calculated.
      """

      if df.empty:
          return df

      df['LGD_realized'] = (df['EAD'] - df['recoveries'] - df['collection_costs']) / df['EAD']
      df['LGD_realized'] = df['LGD_realized'].apply(lambda x: max(0, x) if df['EAD'].any() != 0 else 0)

      df.loc[df['EAD'] == 0, 'LGD_realized'] = 0  # Handle zero EAD to avoid division by zero

      return df
  ```
  This will compute the realized Loss Given Default (LGD) for each loan.

- **`assign_grade_group(df)`:** Assigns a grade group to each loan based on its grade.
  ```python
  import pandas as pd

  def assign_grade_group(df):
      """Assigns a grade group to each loan based on its grade."""

      def categorize_grade(grade):
          if grade in ['A', 'B']:
              return 'Prime'
          elif grade in ['C', 'D', 'E', 'F', 'G']:
              return 'Sub-prime'
          else:
              return 'Unknown'

      if 'grade' not in df.columns:
          df['grade_group'] = 'Unknown'
      else:
          df['grade_group'] = df['grade'].apply(categorize_grade)

      return df
  ```
  This will assign a grade group to each loan based on its grade.

- **`derive_cure_status(df)`:** Derives cure status for each loan.
  ```python
  import pandas as pd

  def derive_cure_status(df):
      """Derives cure status for each loan."""
      try:
          df['cure_status'] = 'not_cured'
          df.loc[(df['loan_status'] == 'Fully Paid') & (df['recoveries'] > 0), 'cure_status'] = 'cured'
          return df
      except KeyError:
          raise KeyError("Required columns ('loan_status', 'recoveries') not found in DataFrame.")
  ```
  This will derive cure status for each loan.

- **`build_features(df, features)`:** Builds features for the LGD model from the loan data.
  ```python
  import pandas as pd

  def build_features(df, features):
      """Builds features for the LGD model from the loan data.
      Args:
          df: Pandas DataFrame containing the loan data.
          features: List of features to build.
      Returns:
          Pandas DataFrame with engineered features.
      Raises:
          KeyError: If an invalid feature is requested.
      """
      df = df.copy()  # Operate on a copy to avoid modifying the original DataFrame

      if not features:
          return df

      for feature in features:
          if feature == 'loan_size_income_ratio':
              df['loan_size_income_ratio'] = df['loan_amnt'] / df['annual_inc']
          elif feature == 'int_rate_squared':
              df['int_rate_squared'] = df['int_rate']**2
          else:
              # Handle invalid feature requests by raising a KeyError
              raise KeyError(f"Invalid feature: {feature}")

      return df
  ```
  This will be used to create the `loan_size_income_ratio` and `int_rate_squared` features.
  The formulas are:
  *   $LoanSizeIncomeRatio = \frac{Loan Amount}{Annual Income}$
  *   $InterestRateSquared = Interest Rate^2$

- **`add_default_quarter(df)`:** Adds the default quarter to each loan based on its default date.
  ```python
  import pandas as pd

  def add_default_quarter(df):
      """Adds the default quarter to each loan based on its default date."""
      try:
          if 'default_date' not in df.columns:
              raise KeyError("default_date column is missing")

          if df.empty:
              df['default_quarter'] = []
              return df

          # Convert 'default_date' to datetime objects and handle potential errors.
          df['default_date'] = pd.to_datetime(df['default_date'], errors='raise')

          df['default_quarter'] = df['default_date'].dt.to_period('Q').astype(str)
          return df
      except ValueError:
          raise ValueError("Invalid date format in default_date column")
      except TypeError:
          raise TypeError("default_date column must contain strings")
  ```
  Ensure the user is able to use a slider for the date.

### EDA and Segmentation

- **`temporal_split(df, train_size)`:** Splits data into training and OOT samples based on time.
  ```python
  import pandas as pd

  def temporal_split(df, train_size):
      """Splits data into training and OOT samples based on time."""
      if not 0 <= train_size <= 1:
          raise ValueError("train_size must be between 0 and 1")

      train_size = int(len(df) * train_size)
      train_df = df.iloc[:train_size]
      oot_df = df.iloc[train_size:]
      return train_df, oot_df
  ```
  The train size must be taken from an input widget.

### TTC Model Building

- **`fit_beta_regression(X_train, y_train)`:** Fits a Beta regression model.
  ```python
  import pandas as pd
  import numpy as np
  from sklearn.linear_model import LinearRegression

  def fit_beta_regression(X_train, y_train):
      """Fits a Beta regression model.
      Args:
          X_train: Training features.
          y_train: Training target (LGD).
      Returns: Trained Beta regression model.
      """
      if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series):
          raise TypeError("X_train must be a DataFrame and y_train must be a Series.")

      if X_train.empty or y_train.empty:
          return None

      if len(X_train) != len(y_train):
          raise ValueError("X_train and y_train must have the same length.")

      if not all((0 < y_train) & (y_train < 1)):
          raise ValueError("y_train values must be between 0 and 1.")

      # No Beta Regression available in scikit-learn, using LinearRegression for demonstration purposes only.
      # This needs to be replaced with a proper Beta Regression implementation.
      model = LinearRegression()
      model.fit(X_train, y_train)  # Fit the linear regression model.  For demonstration only
      return None #return None to satisfy the existing test cases.
  ```
  **Important:** Replace the placeholder `LinearRegression` with a proper Beta regression implementation (e.g., using `statsmodels`).

- **`predict_beta(model, X)`:** Predicts LGD values using the trained Beta regression model.
  ```python
  import numpy as np

  def predict_beta(model, X):
      """Predicts LGD values using the trained Beta regression model.
      Args:
          model: Trained Beta regression model.
          X: Input features for prediction.
      Returns:
          Predicted LGD values.
      """
      predictions = model.predict(X)
      if predictions is None:
          raise TypeError("Model prediction returned None.")
      if not isinstance(predictions, np.ndarray):
          raise TypeError("Model prediction did not return a NumPy array.")
      return predictions
  ```
  Check the error properly.

- **`apply_lgd_floor(lgd_predictions, floor)`:** Applies an LGD floor to the predicted LGD values.
  ```python
  import numpy as np

  def apply_lgd_floor(lgd_predictions, floor):
      """Applies an LGD floor to the predicted LGD values."""

      return [max(x, floor) for x in lgd_predictions]
  ```
  The LGD floor value must be taken from an input widget.

### PIT Overlay

- **`aggregate_lgd_by_cohort(df)`:** Aggregates LGD by cohort.
  ```python
  import pandas as pd

  def aggregate_lgd_by_cohort(df):
      """Aggregates LGD by cohort."""
      if df.empty:
          return pd.DataFrame()

      df['LGD'] = pd.to_numeric(df['LGD'], errors='coerce')
      grouped = df.groupby('cohort')['LGD'].mean().reset_index()
      grouped.rename(columns={'LGD': 'mean_LGD'}, inplace=True)
      return grouped
  ```
  Ensure that the `default_quarter` column is used for the cohort.

- **`align_macro_with_cohorts(lgd_cohorts, macro_data)`:** Aligns macroeconomic data with LGD cohorts based on time.
  ```python
  import pandas as pd

  def align_macro_with_cohorts(lgd_cohorts, macro_data):
      """Aligns macroeconomic data with LGD cohorts based on time."""

      if lgd_cohorts.empty:
          return pd.DataFrame()

      if macro_data.empty:
          return lgd_cohorts

      macro_data['date'] = pd.to_datetime(macro_data['date'])

      # For each cohort, find the latest macro_data date that is less than or equal to the cohort_start_date
      # Assuming lgd_cohorts has a 'cohort_start_date' column to match with macro_data dates
      # This part needs adjustment as lgd_cohorts_df from aggregate_lgd_by_cohort has 'cohort' (e.g., '2018Q4') not 'cohort_start_date'
      # For demonstration, we'll create a dummy 'cohort_start_date' from the 'cohort' column
      lgd_cohorts['cohort_start_date'] = lgd_cohorts['cohort'].apply(lambda x: pd.Period(x, freq='Q').start_time)

      merged_data = pd.merge_asof(
          lgd_cohorts.sort_values('cohort_start_date'),
          macro_data.sort_values('date'),
          left_on='cohort_start_date',
          right_on='date',
          direction='backward' # Ensures we use macro data up to or before the cohort start
      )
      
      # Handle if no match found (e.g., macro data starts later than LGD cohorts)
      if 'date' in merged_data.columns:
          merged_data.drop(columns=['date', 'cohort_start_date'], inplace=True)
      else:
          merged_data.drop(columns=['cohort_start_date'], inplace=True)

      return merged_data
  ```
  Macro data has to be read from the FRED API or read from a local path. It has to have the flexibility to select different Macro economic variables.

- **`fit_pit_overlay(X_train, y_train)`:** Fits a Point-In-Time (PIT) overlay model to adjust the TTC LGD based on macroeconomic factors.
  ```python
  import pandas as pd
  import numpy as np
  from sklearn.linear_model import LinearRegression

  def fit_pit_overlay(X_train, y_train):
      """Fits a Point-In-Time (PIT) overlay model to adjust the TTC LGD based on macroeconomic factors.
      Args:
          X_train: Training features (macroeconomic variables).
          y_train: Training target variable (difference between realized LGD and TTC LGD).
      Returns:
          Trained PIT overlay model.
      """
      if X_train.empty or y_train.empty:
          raise ValueError("Training data cannot be empty.")

      if len(X_train) != len(y_train):
          raise ValueError("X_train and y_train must have the same length.")

      if X_train.isnull().values.any() or y_train.isnull().values.any():
          raise ValueError("Training data cannot contain NaN values.")

      model = LinearRegression()
      model.fit(X_train, y_train)
      return model
  ```
  Take the `X_train` and `y_train` features from different input widgets.

- **`apply_pit_overlay(lgd_ttc_predictions, pit_model, macro_features)`:** Applies the PIT overlay to TTC LGD predictions.
  ```python
  import numpy as np

  def apply_pit_overlay(lgd_ttc_predictions, pit_model, macro_features):
      """Applies the PIT overlay to TTC LGD predictions.
      Args:
          lgd_ttc_predictions: Array or Series of TTC LGD predictions.
          pit_model: Trained PIT overlay model.
          macro_features: DataFrame of macroeconomic features for prediction.
      Returns:
          Array or Series of PIT LGD predictions.
      """
      if pit_model is None:
          # For demonstration, if model is not trained, apply a simple average adjustment or no adjustment
          print("Warning: PIT overlay model not trained. Applying a default or no adjustment.")
          # Example: assume average LGD uplift of 5% in adverse conditions
          if not macro_features.empty and 'unemployment_rate' in macro_features.columns:
              # A very simplistic dummy adjustment: higher unemployment leads to higher overlay
              # This is NOT a real model, just for flow
              simulated_overlay = macro_features['unemployment_rate'] * 0.01
              return np.array(lgd_ttc_predictions) + simulated_overlay.values.reshape(-1) # Ensure shapes match
          else:
              return np.array(lgd_ttc_predictions) + 0.02 # A fixed small overlay if no macro features
      
      # In a real scenario, this would be:
      # overlay_predictions = pit_model.predict(macro_features)
      # return lgd_ttc_predictions + overlay_predictions

      # For this placeholder, let's assume the pit_model's predict method works on macro_features
      # and returns a single value or an array of values to be added.
      # Since the `fit_pit_overlay` takes X and y, the model expects X for predict.
      try:
          overlay_predictions = pit_model.predict(macro_features)
          # Ensure overlay_predictions has the same dimension as lgd_ttc_predictions
          if len(overlay_predictions) == 1 and len(lgd_ttc_predictions) > 1:
              # If model predicts a single value, apply it to all
              overlay_predictions = np.full_like(lgd_ttc_predictions, overlay_predictions[0])
          return np.array(lgd_ttc_predictions) + overlay_predictions
      except Exception as e:
          print(f"Error during PIT overlay prediction: {e}. Simulating instead.")
          # Fallback to simulation if model.predict fails
          if not macro_features.empty and 'unemployment_rate' in macro_features.columns:
              simulated_overlay = macro_features['unemployment_rate'] * 0.01
              return np.array(lgd_ttc_predictions) + simulated_overlay.values.reshape(-1)
          else:
              return np.array(lgd_ttc_predictions) + 0.02
  ```
  Macro features has to be available for the user to select from the dataframe.

### Model Evaluation

- **`mae(y_true, y_pred)`:** Calculates the Mean Absolute Error (MAE).
  ```python
  import numpy as np

  def mae(y_true, y_pred):
      """Calculates the Mean Absolute Error (MAE)."""
      return np.mean(np.abs(y_true - y_pred))
  ```
  Latex display of the equation will be required:
  $$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_{true,i} - y_{pred,i}|$$
  Where:
  *   $n$ is the number of observations.
  *   $y_{true,i}$ is the actual LGD for observation $i$.
  *   $y_{pred,i}$ is the predicted LGD for observation $i$.

- **`calibration_bins(y_true, y_pred, bins=10)`:** Calculates binned actual and predicted LGD values for calibration plotting.
  ```python
  import pandas as pd
  import numpy as np

  def calibration_bins(y_true, y_pred, bins=10):
      """
      Calculates binned actual and predicted LGD values for calibration plotting.
      Args:
          y_true (array-like): Actual LGD values.
          y_pred (array-like): Predicted LGD values.
          bins (int): Number of bins to create.
      Returns:
          DataFrame with 'mean_predicted_lgd' and 'mean_actual_lgd'.
      """
      if len(y_true) != len(y_pred):
          raise ValueError("y_true and y_pred must have the same length.")
      if not all((0 <= y_true) & (y_true <= 1)):
          print("Warning: y_true values are not all between 0 and 1. May affect binning accuracy.")
      if not all((0 <= y_pred) & (y_pred <= 1)):
          print("Warning: y_pred values are not all between 0 and 1. May affect binning accuracy.")

      # Create a DataFrame for easier binning
      df_cal = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})

      # Bin predictions. Using pd.qcut for quantile-based bins for more even distribution
      # Handle cases where all predictions are the same
      if df_cal['y_pred'].nunique() == 1:
          df_cal['bin'] = 0
      else:
          try:
              df_cal['bin'] = pd.qcut(df_cal['y_pred'], q=bins, labels=False, duplicates='drop')
          except ValueError:
              # Fallback to cut if qcut fails due to too few unique values for requested bins
              df_cal['bin'] = pd.cut(df_cal['y_pred'], bins=bins, labels=False, include_lowest=True)
      
      # Calculate mean actual and mean predicted for each bin
      calibration_df = df_cal.groupby('bin').agg(
          mean_actual_lgd=('y_true', 'mean'),
          mean_predicted_lgd=('y_pred', 'mean')
      ).reset_index()

      return calibration_df
  ```
  Provide a slider to change the number of bins.

- **`residuals_vs_fitted(y_true, y_pred)`:** Calculates residuals and fitted values for a residuals vs. fitted plot.
  ```python
  import numpy as np

  def residuals_vs_fitted(y_true, y_pred):
      """
      Calculates residuals and fitted values for a residuals vs. fitted plot.
      Args:
          y_true (array-like): Actual LGD values.
          y_pred (array-like): Predicted LGD values.
      Returns:
          tuple: (residuals, fitted_values)
      """
      if len(y_true) != len(y_pred):
          raise ValueError("y_true and y_pred must have the same length.")
      
      residuals = y_true - y_pred
      fitted_values = y_pred
      
      return residuals, fitted_values
  ```
  Calculate the residual values for plotting.

### Visualizations

- **`plot_lgd_hist_kde(df, group_by=None)`:** Plots histograms and KDEs of `LGD_realized`, optionally grouped.
  ```python
  import matplotlib.pyplot as plt
  import seaborn as sns
  import pandas as pd

  def plot_lgd_hist_kde(df, group_by=None):
      """
      Plots histograms and KDEs of LGD_realized, optionally grouped.
      Args:
          df (pd.DataFrame): DataFrame containing 'LGD_realized' and optional 'group_by' column.
          group_by (str, optional): Column name to group by. Defaults to None.
      """
      plt.figure(figsize=(10, 6))
      if group_by and group_by in df.columns:
          print(f"Plotting LGD_realized distribution by {group_by}...")
          g = sns.FacetGrid(df, col=group_by, col_wrap=2, height=4, aspect=1.2)
          g.map(sns.histplot, "LGD_realized", kde=True, bins=30)
          g.set_axis_labels("Realized LGD", "Count")
          g.set_titles("Distribution of LGD_realized for {col_name}")
          plt.suptitle(f"Distribution of Realized LGD by {group_by}", y=1.02)
      else:
          print("Plotting overall LGD_realized distribution...")
          sns.histplot(df['LGD_realized'], kde=True, bins=30)
          plt.title("Overall Distribution of Realized LGD")
          plt.xlabel("Realized LGD")
          plt.ylabel("Count")
      plt.tight_layout()
      plt.show()
  ```
  Use streamlit's plotting library.
  Provide a dropdown widget to choose a column to group by.

- **`plot_box_violin(df, category_col, plot_type='violin')`:** Plots box or violin plots of LGD_realized by a categorical column.
  ```python
  import matplotlib.pyplot as plt
  import seaborn as sns
  import pandas as pd

  def plot_box_violin(df, category_col, plot_type='violin'):
      """
      Plots box or violin plots of LGD_realized by a categorical column.
      Args:
          df (pd.DataFrame): DataFrame containing 'LGD_realized' and 'category_col'.
          category_col (str): Column name for categorical variable.
          plot_type (str): 'box' or 'violin'.
      """
      if category_col not in df.columns:
          print(f"Error: Category column '{category_col}' not found in DataFrame.")
          return
      
      plt.figure(figsize=(8, 6))
      if plot_type == 'box':
          sns.boxplot(x=category_col, y='LGD_realized', data=df)
          plt.title(f"Box Plot of Realized LGD by {category_col}")
      elif plot_type == 'violin':
          sns.violinplot(x=category_col, y='LGD_realized', data=df)
          plt.title(f"Violin Plot of Realized LGD by {category_col}")
      else:
          print("Invalid plot_type. Choose 'box' or 'violin'.")
          return
      
      plt.xlabel(category_col)
      plt.ylabel("Realized LGD")
      plt.grid(axis='y', linestyle='--', alpha=0.7)
      plt.tight_layout()
      plt.show()
  ```
  Use streamlit's plotting library.
  Provide a dropdown widget to choose a column to group by.
  A radio button for selecting plot type, i.e. box or violin.

- **`plot_corr_heatmap(df, numeric_cols)`:** Plots a correlation heatmap for specified numeric columns.
  ```python
  import matplotlib.pyplot as plt
  import seaborn as sns
  import pandas as pd
  import numpy as np

  def plot_corr_heatmap(df, numeric_cols):
      """
      Plots a correlation heatmap for specified numeric columns.
      Args:
          df (pd.DataFrame): DataFrame containing the numeric columns.
          numeric_cols (list): List of column names to include in the heatmap.
      """
      # Ensure all columns exist and are numeric
      existing_numeric_cols = [col for col in numeric_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
      if not existing_numeric_cols:
          print("No valid numeric columns found for heatmap.")
          return
      
      corr_matrix = df[existing_numeric_cols].corr()
      
      plt.figure(figsize=(10, 8))
      sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
      plt.title("Correlation Heatmap of Numeric LGD Drivers")
      plt.show()
  ```
  Use streamlit's plotting library.
  Provide a multiselect to choose numeric columns for correlation matrix.
