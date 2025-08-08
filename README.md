# QuLab: Loss Given Default (LGD) Model Exploration and Visualization

## Project Title and Description

QuLab is a Streamlit application designed to explore and visualize Loss Given Default (LGD) models. This application provides a comprehensive environment for understanding the components of LGD models, developing skills in data handling and feature engineering, applying statistical modeling techniques, incorporating macroeconomic factors, and evaluating model performance for deployment. The primary focus is on Through-The-Cycle (TTC) and Point-In-Time (PIT) LGD approaches.

## Features

*   **Data Ingestion:**
    *   Upload LendingClub loan data from CSV files.
    *   Fetch default LendingClub data directly from a specified URL (2018Q4).
    *   Error handling for file uploads and network requests.

*   **Feature Engineering:**
    *   Filter defaulted loans based on loan status (e.g., "Charged Off").
    *   Assemble recovery cashflows from defaulted loans.
    *   Compute Exposure at Default (EAD) for each loan.
    *   Calculate realized Loss Given Default (LGD).
    *   Generate features such as `loan_size_income_ratio` and `int_rate_squared`.
    *   Categorize loan grades into `Prime` and `Sub-prime`.
    *   Derive `cure_status` to indicate if defaulted loans were eventually fully paid.
    *   Add `default_quarter` feature to represent the quarter when a loan defaulted.

*   **EDA and Segmentation:**
    *   Perform a temporal split of data into training and out-of-time (OOT) sets.
    *   Visualize the distribution of `LGD_realized` using histograms and KDE plots.
    *   Explore `LGD_realized` across different categories using box or violin plots.
    *   Generate a correlation heatmap of numeric features to understand their relationships.

*   **TTC Model Building:**
    *   Build a Through-The-Cycle (TTC) LGD model using Beta Regression.
    *   Select features to train the LGD model.
    *   Apply an LGD floor value.
    *   Visualize model performance with predicted vs. actual plots, calibration curves, and residual plots.

*   **PIT Overlay:**  *(Currently placeholder - to be implemented)*
    *   (Future Feature) Incorporate macroeconomic factors into LGD models (Point-In-Time adjustment).
    *   (Future Feature) Overlay Point-In-Time (PIT) factors on the TTC model.

*   **Model Evaluation:** *(Currently placeholder - to be implemented)*
    *   (Future Feature) Evaluate model performance metrics.
    *   (Future Feature) Compare TTC and PIT LGD models.

*   **Model Export:** *(Currently placeholder - to be implemented)*
    *   (Future Feature) Export the trained LGD model for deployment.
    *   (Future Feature) Generate model artifacts (e.g., documentation).

## Getting Started

### Prerequisites

*   Python 3.7+
*   pip package manager
*   Git (Optional, for cloning the repository)

### Installation

1.  **Clone the repository (Optional):**

    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment (Recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install streamlit pandas numpy plotly statsmodels requests
    ```

## Usage

1.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

2.  **Access the application in your browser:**

    Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Using the Application:**

    *   Use the navigation sidebar to access different sections of the application.
    *   Follow the instructions and prompts within each section to load data, perform feature engineering, explore data, build models, and evaluate performance.
    *   Start with the "Data Ingestion" page to load your dataset.
    *   Proceed to "Feature Engineering" to preprocess your data.
    *   Use "EDA and Segmentation" to analyze the preprocessed data and split it for training.
    *   Build your LGD model in "TTC Model Building".
    *   Future features like "PIT Overlay," "Model Evaluation," and "Model Export" will further enhance the application's capabilities.

## Project Structure

```
QuLab/
├── app.py                        # Main Streamlit application file
├── application_pages/            # Directory containing individual application pages
│   ├── data_ingestion.py        # Data Ingestion page
│   ├── feature_engineering.py   # Feature Engineering page
│   ├── eda_segmentation.py      # EDA and Segmentation page
│   ├── ttc_model_building.py    # TTC Model Building page
│   ├── pit_overlay.py           # PIT Overlay page (Currently placeholder)
│   ├── model_evaluation.py      # Model Evaluation page (Currently placeholder)
│   └── model_export.py          # Model Export page (Currently placeholder)
├── README.md                     # This file
└── venv/                         # (Optional) Virtual environment directory (not usually committed)

```

## Technology Stack

*   **Streamlit:**  Used for building the interactive web application.
*   **Pandas:**  For data manipulation and analysis.
*   **NumPy:** For numerical computations.
*   **Plotly:** For creating interactive plots and visualizations.
*   **Statsmodels:**  For statistical modeling (Beta Regression).
*   **Requests:**  For fetching default dataset.
*   **io, zipfile:** For handling zipped data.

## Contributing

Contributions are welcome! To contribute to this project, please follow these guidelines:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with clear, descriptive messages.
4.  Test your changes thoroughly.
5.  Submit a pull request to the main branch.

## License

This project is licensed under the [MIT License](LICENSE) - see the `LICENSE` file for details.

## Contact

For questions, issues, or contributions, please contact:

*   [Your Name/Organization]
*   [Your Email/Contact Link]
