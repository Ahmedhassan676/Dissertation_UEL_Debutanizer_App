# Machine Learning Model Predictor & Analyzer

This is a comprehensive Streamlit web application designed for end-to-end machine learning workflows on time-series and tabular data. It allows users to upload a CSV file and perform Exploratory Data Analysis (EDA), anomaly detection, model training, and even process optimization without writing any code.



---

## 1. Core Functionalities & Capabilities

The application is structured into three main phases, selectable from the sidebar, each offering a suite of powerful tools.

### Phase 1: Exploratory Data Analysis (EDA)

This phase is dedicated to understanding the uploaded dataset.

-   **Basic EDA**:
    -   **Dataframe Info**: A summary of columns, data types, and non-null values (`df.info()`).
    -   **Descriptive Statistics**: Key statistical details for all numerical columns (`df.describe()`).
    -   **Correlation Heatmap**: A visual matrix showing the Pearson correlation between all numerical variables.
    -   **Variable Trends**: Line plots for each numerical column to visualize trends over time or index.
    -   **Pairwise Plots**: A grid of scatterplots for every pair of numerical variables to spot relationships.

-   **Lag Correlation Analysis**:
    -   Investigates the time-delayed relationship between input features and a selected target variable.
    -   Calculates and visualizes **Pearson, Kendall, and Spearman** correlations across a user-defined range of lags.
    -   Automatically identifies the "best" lag for each feature (the lag with the highest absolute correlation).
    -   Generates and allows downloading a **lag-optimized dataset**, where each feature is shifted by its optimal lag, creating a powerful feature-engineered dataset for modeling.

-   **3D Feature Visualization**:
    -   Creates interactive 3D scatter plots using Plotly to explore the relationships between three features, with a fourth used for color-coding.

### Phase 2: Anomaly Detection

This phase provides a robust framework for identifying and handling outliers in the data.

-   **Multi-Method Approach**: Employs an ensemble of seven different anomaly detection techniques:
    1.  **Z-Score**: Identifies points that deviate significantly from the mean.
    2.  **Rolling Statistics**: Uses a rolling window to detect anomalies based on local mean and standard deviation.
    3.  **Isolation Forest**: An unsupervised algorithm that isolates anomalies by their path length in a random forest.
    4.  **Local Outlier Factor (LOF)**: A density-based method that identifies outliers based on their local neighborhood.
    5.  **PCA Reconstruction**: Flags anomalies based on high reconstruction error after dimensionality reduction with PCA.
    6.  **PLS Residuals**: Uses Partial Least Squares regression to model the target and flags points with high prediction error.
    7.  **Linear Regression Residuals**: Similar to PLS, but uses a simpler linear model to find points that don't fit the general trend.

-   **Voting System**: Aggregates the results from all selected methods. A data point is flagged as an anomaly if it surpasses a user-defined "vote" threshold.

-   **Visualization & Download**:
    -   Provides detailed plots showing which points were flagged by each individual method.
    -   A combined plot highlights the final set of anomalies based on the voting threshold.
    -   Allows users to download a **cleaned dataset** with the identified anomalous rows removed.

### Phase 3: Regression Models

This is the core modeling phase, offering a wide variety of regression and time-series models.

-   **Model Categories**:
    -   **Time Series Models**:
        -   `Simple LSTM`: A standard stacked LSTM network.
        -   `LSTM Encoder-Decoder with Attention`: An advanced sequence-to-sequence model for multi-step forecasting.
        -   `LSTM Encoder-Decoder with Attention + 1D CNN`: Adds a convolutional layer.
        -   `Hankel DMD with Control`: A physics-informed model (Dynamic Mode Decomposition) for systems with control inputs.
    -   **Lagged Regression Models**:
        -   Includes `Random Forest`, `Gradient Boosting`, `XGBoost`, `Linear Regression`, `SVR`, `Principal Component Regression (PCR)`, and `MLP`.
        -   These models are automatically trained and evaluated across a range of user-specified lags to find the optimal lag structure.
    -   **Multi-Target Models**:
        -   Includes multi-output versions of `Linear Regression`, `Random Forest`, `XGBoost`, and `MLP`.
        -   Capable of predicting multiple target variables simultaneously.

-   **Key Modeling Features**:
    -   **Hyperparameter Tuning**: Integrated `GridSearchCV` allows for exhaustive searching of the best model parameters.
    -   **Detailed Evaluation**: Generates comprehensive evaluation plots including Predicted vs. Actual, Residual Distribution, and Time-Series comparisons.
    -   **Feature Importance**: Automatically plots feature importances (for tree-based models) or coefficients (for linear models) to provide model interpretability.
    -   **Process Optimizer (for Multi-Target Models)**:
        -   After training a multi-target model, a "Process Optimizer" tab appears.
        -   It uses the `nevergrad` library to run an optimization algorithm.
        -   Users can select a target to maximize or minimize, and the optimizer will find the best input variable settings to achieve that goal.
        -   Supports batch optimization on multiple data points and provides rich visualizations of the results.

---

## 2. Configuration and Setup

Follow these steps to set up and run the application on your local machine.

### Prerequisites

-   **Python**: Version 3.9 or higher.
-   **Package Manager**: `pip` or `conda`.

### Installation Steps

1.  **Clone the Repository (Optional)**
    If you have the project in a git repository, clone it. Otherwise, just navigate to the project directory.
    ```bash
    git clone https://github.com/Ahmedhassan676/Dissertation_UEL_Debutanizer_App
    cd Dissertation_UEL_Debutanizer_App
    ```

2.  **Create a Virtual Environment**
    It is highly recommended to use a virtual environment to avoid conflicts with other Python projects.

    *   **Using `venv` (standard with Python):**
        ```bash
        # Create the environment
        python -m venv venv

        # Activate the environment
        # On Windows
        .\venv\Scripts\activate
        # On macOS/Linux
        source venv/bin/activate
        ```

    *   **Using `conda`:**
        ```bash
        # Create and activate the environment
        conda create --name ml_app python=3.9
        conda activate ml_app
        ```

3.  **Install Required Libraries**
    The required libraries and their versions are listed below. You can install them all by creating a `requirements.txt` file with the content below and running a single command.

    Create a file named `requirements.txt` in your project directory and paste the following content into it:

    ```
    streamlit==1.33.0
    pandas==2.1.4
    numpy==1.26.4
    matplotlib==3.8.0
    tensorflow==2.16.1
    scikit-learn==1.3.2
    xgboost==2.0.3
    seaborn==0.13.2
    statsmodels==0.14.1
    plotly==5.18.0
    bayesian-optimization==1.4.3
    nevergrad==1.0.1
    scipy==1.13.0
    pydmd==0.4.1
    ```

    Now, install these packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

    *(Note: `tensorflow` installation can sometimes be complex, especially with GPU support. If you encounter issues, please refer to the official TensorFlow installation guide.)*

### How to Run the App

Once the setup is complete, you can run the Streamlit application from your terminal or Anaconda Prompt.

1.  Make sure your virtual environment is activated.
2.  Navigate to the directory containing the `Models.py` file.
3.  Run the following command:

    ```bash
    streamlit run Models.py
    ```

4.  The application will automatically open in a new tab in your default web browser. You can also access it at the local URL provided in the terminal (usually `http://localhost:8501`).


---

