# QuLab: Loss Given Default (LGD) Modeling Application

## Project Title and Description

**QuLab** is a Streamlit web application designed for exploring and building Loss Given Default (LGD) models. LGD represents the expected loss when a borrower defaults on a loan. This application provides a step-by-step workflow for data ingestion, feature engineering, exploratory data analysis (EDA), Through-The-Cycle (TTC) model building, Point-In-Time (PIT) overlay implementation, model evaluation, and model export.  The application aims to provide an interactive environment for understanding and experimenting with LGD modeling techniques.

## Features

*   **Data Ingestion:**
    *   Upload LendingClub dataset in CSV format.
    *   Data validation and error handling.
*   **Feature Engineering:**
    *   Filter data based on loan status.
    *   Input discount rate for present value calculations.
    *   Preview of filtered data.
*   **EDA and Segmentation:**
    *   Visualize data distributions using histograms.
    *   Interactive data filtering using sliders based on selected features
    *   Generate box plots to explore feature relationships.
*   **TTC Model Building:**
    *   Prepare data for TTC LGD model: filtering defaults, assembling recovery cashflows, computing Exposure at Default (EAD), present value of cashflows, and realized LGD.
    *   Feature selection for model training.
    *   Train a Beta regression model.
    *   Apply an LGD floor to predictions.
    *   Visualize model performance with scatter plots, calibration curves, and residual plots.
*   **PIT Overlay:**
    *   Incorporate macroeconomic factors (e.g., unemployment rate) for Point-In-Time (PIT) adjustments to TTC model.
    *   Aggregate LGD by cohort.
    *   Align macroeconomic data with LGD cohorts.
    *   Visualize LGD and unemployment rate trends.
    *   Conduct scenario analysis using stress testing.
*   **Model Evaluation:**
    *   Calculate Mean Absolute Error (MAE).
    *   Visualize model performance metrics and plots.
*   **Model Export:**
    *   Download the test dataset.
    *   (Future implementation) Download the trained TTC model.

## Getting Started

### Prerequisites

*   Python 3.7+
*   pip
*   Virtual environment (recommended)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    ```

    *   Activate the virtual environment:
        *   On Windows: `venv\Scripts\activate`
        *   On macOS/Linux: `source venv/bin/activate`

3.  **Install the required packages:**

    ```bash
    pip install streamlit pandas scikit-learn plotly requests
    ```

## Usage

1.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

    This will open the application in your default web browser.

2.  **Navigate through the application using the sidebar:**

    *   **Data Ingestion:** Upload your LendingClub dataset (CSV file).
    *   **Feature Engineering:** Filter the data, select relevant features, and set the discount rate.
    *   **EDA and Segmentation:** Explore the data visually and identify potential segments.
    *   **TTC Model Building:** Train the Through-The-Cycle (TTC) LGD model.
    *   **PIT Overlay:** Incorporate macroeconomic factors for Point-In-Time (PIT) adjustments.
    *   **Model Evaluation:** Evaluate the performance of your LGD model.
    *   **Model Export:** Download the model and test dataset.

## Project Structure

```
QuLab/
├── app.py                              # Main Streamlit application file
├── application_pages/
│   ├── data_ingestion.py           # Data ingestion module
│   ├── feature_engineering.py      # Feature engineering module
│   ├── eda_segmentation.py         # EDA and segmentation module
│   ├── ttc_model_building.py       # TTC model building module
│   ├── pit_overlay.py              # PIT overlay module
│   ├── model_evaluation.py         # Model evaluation module
│   ├── model_export.py             # Model export module
│   └── utils.py                    # Utility functions
├── README.md                           # This file
└── venv/                               # Virtual environment (optional)
```

## Technology Stack

*   **Streamlit:** Web application framework
*   **Pandas:** Data manipulation and analysis
*   **Scikit-learn:** Machine learning library
*   **Plotly:** Interactive data visualization
*   **Requests:** HTTP library for fetching data

## Contributing

Contributions are welcome! Please follow these guidelines:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with clear, descriptive messages.
4.  Submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions or issues, please contact:

*   [Your Name/Organization]
*   [Your Email/Website]
