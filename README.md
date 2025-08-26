# SkyCast: Weather Prediction Web App

SkyCast is a comprehensive weather prediction web application built with Python, Streamlit, and scikit-learn. The app allows users to upload weather datasets, explore and visualize the data, train multiple machine learning models, and make weather forecasts. It provides an interactive interface for data analysis, model comparison, and prediction visualization with features like data cleaning, correlation analysis, time series plotting, geographic location mapping, and model performance evaluation.

## Features

*   **Data Upload and Cleaning:** Upload your own weather data in CSV format or use the sample dataset. The app provides tools to clean and preprocess the data for analysis.
*   **Exploratory Data Analysis (EDA):** Interactively explore the dataset with visualizations like time series plots, correlation heatmaps, and distribution plots.
*   **Machine Learning Model Training:** Train multiple regression models (Linear Regression, Random Forest, Gradient Boosting) to predict weather phenomena.
*   **Model Evaluation and Comparison:** Evaluate the performance of trained models using various metrics (MAE, RMSE, R-squared) and compare them to select the best-performing model.
*   **Weather Prediction:** Make future weather predictions using the trained models and visualize the results.
*   **Interactive UI:** A user-friendly and interactive web interface built with Streamlit.

## Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/SkyCast.git
    cd SkyCast
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

## Usage

1.  **Upload Data:**
    *   Launch the application.
    *   On the "Upload Data" page, you can either load the sample weather data or upload your own CSV file.
    *   Click the "Clean Dataset" button to preprocess the data.

2.  **Explore Data:**
    *   Navigate to the "Explore Data" page to visualize the dataset.
    *   You can view the data in an interactive table and generate various plots.

3.  **Train Model:**
    *   Go to the "Train Model" page.
    *   Select the target variable and feature columns.
    *   Choose the machine learning models you want to train.
    *   Click the "Train Models" button to start the training process.
    *   The model performance metrics will be displayed after training.

4.  **Predict Weather:**
    *   Navigate to the "Predict Weather" page.
    *   Select a trained model and specify the number of days to predict.
    *   You can either use the latest data from your dataset or manually input the weather parameters.
    *   Click the "Generate Predictions" button to see the forecast.

## Data

The application comes with a sample weather dataset (`data/sample_weather_data.csv`). You can also use your own data, as long as it is in a CSV format and contains relevant weather-related features.

## Model

The application uses the following machine learning models from scikit-learn for regression tasks:

*   **Linear Regression:** A simple and interpretable linear model.
*   **Random Forest Regressor:** An ensemble model that uses multiple decision trees to improve prediction accuracy.
*   **Gradient Boosting Regressor:** An ensemble model that builds trees one at a time, where each new tree helps to correct errors made by previously trained trees.

The models are trained on the preprocessed data, and their performance is evaluated using standard regression metrics.

## Technologies Used

*   **Python:** The core programming language.
*   **Streamlit:** For building the interactive web application.
*   **Pandas:** For data manipulation and analysis.
*   **Scikit-learn:** For machine learning model training and evaluation.
*   **Plotly:** For creating interactive visualizations.
*   **Joblib:** For saving and loading trained models.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.
