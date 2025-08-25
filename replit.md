# Weather Prediction Web App

## Overview

This is a comprehensive weather prediction web application built with Python, Streamlit, and scikit-learn. The app allows users to upload weather datasets, explore and visualize the data, train multiple machine learning models, and make weather forecasts. It provides an interactive interface for data analysis, model comparison, and prediction visualization with features like data cleaning, correlation analysis, time series plotting, geographic location mapping, and model performance evaluation.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web framework for creating interactive web applications
- **Layout**: Wide layout with sidebar navigation for different app sections
- **Components**: Multi-page application with sections for data upload, exploration, model training, and predictions
- **State Management**: Streamlit session state for persisting data and models across user interactions
- **User Interface**: Clean, interactive UI with progress bars, file uploads, and dynamic visualizations

### Backend Architecture
- **Modular Design**: Separated into utility modules for different functionalities:
  - `data_processing.py`: Handles data cleaning, normalization, and validation
  - `ml_models.py`: Manages machine learning model training and evaluation
  - `visualizations.py`: Creates interactive charts and plots
- **Data Processing**: Automated data cleaning with missing value handling, duplicate removal, and data type conversion
- **Model Training**: Support for multiple ML algorithms (Linear Regression, Random Forest, Gradient Boosting)
- **Prediction Engine**: Forecasting capabilities with confidence intervals and model persistence

### Data Processing Pipeline
- **Input Validation**: CSV file upload with automatic data type detection
- **Data Cleaning**: Forward/backward fill for missing values, outlier detection, and normalization
- **Feature Engineering**: Automatic identification of numeric columns and date parsing
- **Data Splitting**: Train/test split with configurable ratios for model evaluation

### Machine Learning Framework
- **Model Selection**: Multiple algorithm support with scikit-learn integration
- **Training Pipeline**: Automated training with cross-validation and hyperparameter tuning
- **Evaluation Metrics**: MAE, RMSE, and R² score calculations for model comparison
- **Model Persistence**: Save/load functionality using joblib for trained models

### Visualization System
- **Interactive Charts**: Plotly-based visualizations for time series, correlations, and distributions
- **Geographic Mapping**: Interactive weather station location maps with color-coded weather parameters
- **Real-time Updates**: Dynamic chart generation based on user selections
- **Export Capabilities**: Download functionality for predictions and visualizations
- **Responsive Design**: Charts adapt to different screen sizes and data volumes
- **Location Intelligence**: Map-based visualization showing weather patterns across geographic coordinates

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework for the user interface
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing and array operations
- **scikit-learn**: Machine learning algorithms and evaluation metrics

### Visualization Libraries
- **plotly**: Interactive plotting library for charts and graphs
- **matplotlib**: Static plotting for additional visualization options
- **seaborn**: Statistical data visualization

### Utility Libraries
- **joblib**: Model serialization and persistence
- **warnings**: Warning message filtering for cleaner output

### Data Processing
- **sklearn.preprocessing**: Data scaling and normalization
- **sklearn.model_selection**: Train/test splitting and cross-validation
- **sklearn.metrics**: Model evaluation metrics

### File System
- **os**: Operating system interface for file operations
- **datetime**: Date and time manipulation for forecasting features