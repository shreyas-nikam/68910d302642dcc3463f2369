# QuLab: Loss Given Default (LGD) Modeling with Streamlit

## Project Title and Description

**QuLab** is an interactive Streamlit application designed to guide users through the process of building and evaluating Loss Given Default (LGD) models. LGD is a critical component of credit risk management, representing the proportion of an exposure lost when a borrower defaults. This application explores two primary LGD modeling approaches: Through-The-Cycle (TTC) and Point-In-Time (PIT) models, enabling users to understand their components, apply statistical techniques, and integrate macroeconomic factors.

## Features

QuLab provides a comprehensive suite of features for LGD modeling:

*   **Data Ingestion:** Upload LendingClub loan data from a CSV file or load a sample dataset directly within the application.
*   **Feature Engineering:** Prepare raw loan data by filtering for defaulted loans, assembling recovery cashflows, computing Exposure at Default (EAD), and deriving the Realized LGD.
*   **EDA and Segmentation:** Perform exploratory data analysis (EDA) to understand the distribution of LGD and identify relationships with loan characteristics using interactive visualizations.
*   **TTC Model Building:** Build a Through-The-Cycle (TTC) LGD model using Beta regression to estimate the long-run average LGD. Configure model parameters and evaluate performance using various metrics and plots.
*   **PIT Overlay:** Develop a Point-In-Time (PIT) overlay model to adjust the TTC LGD based on current macroeconomic conditions, enhancing the model's responsiveness to economic cycles.
*   **Model Evaluation:** Evaluate the performance of both TTC and PIT models using metrics like Mean Absolute Error (MAE) and visualize error distributions.
*   **Model Export:** Download trained LGD models and processed data for external use, auditing, or deployment.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

*   **Python 3.7+:**  Ensure you have Python 3.7 or a later version installed.
*   **Pip:** Python package installer.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd QuLab
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**

    *   **On Windows:**

        ```bash
        venv\Scripts\activate
        ```

    *   **On macOS and Linux:**

        ```bash
        source venv/bin/activate
        ```

4.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

    **Note**: A `requirements.txt` file needs to be created including all the necessary packages, using the following command:
    ```bash
    pip freeze > requirements.txt
    ```
    The `requirements.txt` should at least contain:
    ```
    streamlit
    pandas
    numpy
    statsmodels
    plotly
    scikit-learn
    joblib
    requests
    ```

## Usage

1.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

2.  **Access the application:** Open your web browser and navigate to the URL displayed in the terminal (usually `http://localhost:8501`).

3.  **Navigate through the application:** Use the sidebar to select different pages:

    *   **Data Ingestion:** Load your loan data.
    *   **Feature Engineering:** Process and transform the loaded data.
    *   **EDA and Segmentation:** Explore the data through visualizations.
    *   **TTC Model Building:** Configure and train the TTC LGD model.
    *   **PIT Overlay:**  Build the PIT overlay model to adjust the TTC model.
    *   **Model Evaluation:** Evaluate and compare model performance.
    *   **Model Export:** Download the trained models and processed data.

4.  **Follow the on-screen instructions:** Each page provides detailed guidance and interactive elements to guide you through the LGD modeling process.

## Project Structure

```
QuLab/
├── app.py                       # Main Streamlit application file
├── application_pages/         # Directory containing individual page scripts
│   ├── page1.py               # Data Ingestion
│   ├── page2.py               # Feature Engineering
│   ├── page3.py               # EDA and Segmentation
│   ├── page4.py               # TTC Model Building
│   ├── page5.py               # PIT Overlay
│   ├── page6.py               # Model Evaluation
│   ├── page7.py               # Model Export
├── requirements.txt           # List of Python dependencies
├── README.md                  # This file
