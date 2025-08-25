import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import utility modules
from utils.data_processing import DataProcessor
from utils.ml_models import WeatherMLModels
from utils.visualizations import WeatherVisualizations

# Configure page
st.set_page_config(
    page_title="Weather Prediction App",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Initialize utility classes
data_processor = DataProcessor()
ml_models = WeatherMLModels()
visualizations = WeatherVisualizations()

def main():
    # Header
    st.title("🌤️ Weather Prediction Web App")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["Upload Data", "Explore Data", "Train Model", "Predict Weather"]
    )
    
    if page == "Upload Data":
        upload_data_page()
    elif page == "Explore Data":
        explore_data_page()
    elif page == "Train Model":
        train_model_page()
    elif page == "Predict Weather":
        predict_weather_page()

def upload_data_page():
    st.header("📁 Upload Weather Data")
    
    # Sample data option
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Option 1: Use Sample Dataset")
        if st.button("Load Sample Weather Data"):
            try:
                sample_data = pd.read_csv("data/sample_weather_data.csv")
                st.session_state.data = sample_data
                st.success("Sample dataset loaded successfully!")
                st.dataframe(sample_data.head())
            except FileNotFoundError:
                st.error("Sample dataset not found. Please upload your own data.")
    
    with col2:
        st.subheader("Option 2: Upload Your Dataset")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a weather dataset with columns like temperature, humidity, rainfall, etc."
        )
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                st.success("Dataset uploaded successfully!")
                st.dataframe(data.head())
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
    
    # Data cleaning section
    if st.session_state.data is not None:
        st.markdown("---")
        st.subheader("🧹 Data Cleaning")
        
        if st.button("Clean Dataset"):
            with st.spinner("Cleaning dataset..."):
                cleaned_data = data_processor.clean_data(st.session_state.data)
                st.session_state.cleaned_data = cleaned_data
                
                # Show cleaning summary
                st.success("Dataset cleaned successfully!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Shape", f"{st.session_state.data.shape[0]} × {st.session_state.data.shape[1]}")
                with col2:
                    st.metric("Cleaned Shape", f"{cleaned_data.shape[0]} × {cleaned_data.shape[1]}")
                
                # Show data info
                st.subheader("Dataset Information")
                st.dataframe(cleaned_data.describe())

def explore_data_page():
    st.header("📊 Explore Weather Data")
    
    if st.session_state.cleaned_data is None:
        st.warning("Please upload and clean your dataset first!")
        return
    
    data = st.session_state.cleaned_data
    
    # Data overview
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(data))
    with col2:
        st.metric("Features", len(data.columns))
    with col3:
        st.metric("Date Range", f"{len(data)} days")
    with col4:
        st.metric("Missing Values", data.isnull().sum().sum())
    
    # Interactive data table
    st.subheader("Interactive Data Table")
    
    # Filtering options
    col1, col2 = st.columns(2)
    with col1:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_columns:
            filter_column = st.selectbox("Filter by column:", ["None"] + numeric_columns)
    with col2:
        if filter_column != "None":
            min_val = float(data[filter_column].min())
            max_val = float(data[filter_column].max())
            filter_range = st.slider(
                f"Select {filter_column} range:",
                min_val, max_val, (min_val, max_val)
            )
    
    # Apply filters
    filtered_data = data.copy()
    if filter_column != "None":
        filtered_data = filtered_data[
            (filtered_data[filter_column] >= filter_range[0]) & 
            (filtered_data[filter_column] <= filter_range[1])
        ]
    
    st.dataframe(filtered_data, use_container_width=True)
    
    # Visualizations
    st.markdown("---")
    st.subheader("📈 Data Visualizations")
    
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
        "Time Series", "Correlation Heatmap", "Distribution", "Fun Insights"
    ])
    
    with viz_tab1:
        visualizations.plot_time_series(filtered_data)
    
    with viz_tab2:
        visualizations.plot_correlation_heatmap(filtered_data)
    
    with viz_tab3:
        visualizations.plot_distributions(filtered_data)
    
    with viz_tab4:
        visualizations.show_fun_insights(filtered_data)

def train_model_page():
    st.header("🤖 Train Machine Learning Models")
    
    if st.session_state.cleaned_data is None:
        st.warning("Please upload and clean your dataset first!")
        return
    
    data = st.session_state.cleaned_data
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) < 2:
        st.error("Dataset needs at least 2 numeric columns for training!")
        return
    
    # Model configuration
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        target_variable = st.selectbox(
            "Select target variable to predict:",
            numeric_columns,
            help="Choose the weather parameter you want to predict"
        )
    
    with col2:
        feature_columns = st.multiselect(
            "Select feature columns:",
            [col for col in numeric_columns if col != target_variable],
            default=[col for col in numeric_columns if col != target_variable][:5],
            help="Choose the input features for prediction"
        )
    
    if not feature_columns:
        st.error("Please select at least one feature column!")
        return
    
    # Model selection
    st.subheader("Model Selection")
    models_to_train = st.multiselect(
        "Select models to train:",
        ["Linear Regression", "Random Forest", "Gradient Boosting"],
        default=["Random Forest"],
        help="You can train and compare multiple models"
    )
    
    # Training parameters
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
    with col2:
        random_state = st.number_input("Random seed", value=42, min_value=0)
    
    # Train models
    if st.button("🚀 Train Models"):
        if not models_to_train:
            st.error("Please select at least one model to train!")
            return
        
        # Prepare data
        X = data[feature_columns]
        y = data[target_variable]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        trained_models = {}
        model_metrics = {}
        
        for i, model_name in enumerate(models_to_train):
            status_text.text(f"Training {model_name}...")
            progress_bar.progress((i + 1) / len(models_to_train))
            
            try:
                model, metrics = ml_models.train_model(
                    X, y, model_name, test_size, random_state
                )
                trained_models[model_name] = model
                model_metrics[model_name] = metrics
                
                # Save model
                model_path = f"models/{model_name.lower().replace(' ', '_')}_{target_variable}.joblib"
                os.makedirs("models", exist_ok=True)
                joblib.dump(model, model_path)
                
            except Exception as e:
                st.error(f"Error training {model_name}: {str(e)}")
                continue
        
        # Store in session state
        st.session_state.trained_models = trained_models
        st.session_state.model_metrics = model_metrics
        st.session_state.target_variable = target_variable
        st.session_state.feature_columns = feature_columns
        
        status_text.text("Training completed!")
        st.success("Models trained successfully!")
        
        # Display results
        st.subheader("📊 Model Performance")
        
        metrics_df = pd.DataFrame(model_metrics).T
        st.dataframe(metrics_df, use_container_width=True)
        
        # Visualize model comparison
        if len(model_metrics) > 1:
            fig = go.Figure()
            
            metrics_names = list(metrics_df.columns)
            for metric in metrics_names:
                fig.add_trace(go.Bar(
                    name=metric,
                    x=list(metrics_df.index),
                    y=metrics_df[metric],
                ))
            
            fig.update_layout(
                title="Model Performance Comparison",
                xaxis_title="Models",
                yaxis_title="Score",
                barmode='group'
            )
            
            st.plotly_chart(fig, use_container_width=True)

def predict_weather_page():
    st.header("🔮 Weather Prediction")
    
    if not st.session_state.trained_models:
        st.warning("Please train at least one model first!")
        return
    
    # Model selection for prediction
    col1, col2 = st.columns(2)
    with col1:
        selected_model = st.selectbox(
            "Select model for prediction:",
            list(st.session_state.trained_models.keys())
        )
    
    with col2:
        prediction_days = st.number_input(
            "Number of days to predict:",
            min_value=1,
            max_value=30,
            value=7
        )
    
    # Input method selection
    st.subheader("Prediction Input Method")
    input_method = st.radio(
        "Choose input method:",
        ["Use Latest Data Values", "Manual Input"]
    )
    
    if input_method == "Manual Input":
        st.subheader("Enter Weather Parameters")
        input_values = {}
        
        cols = st.columns(len(st.session_state.feature_columns))
        for i, feature in enumerate(st.session_state.feature_columns):
            with cols[i]:
                # Get reasonable default based on data
                feature_data = st.session_state.cleaned_data[feature]
                default_val = float(feature_data.mean())
                min_val = float(feature_data.min())
                max_val = float(feature_data.max())
                
                input_values[feature] = st.number_input(
                    f"{feature}:",
                    value=default_val,
                    min_value=min_val,
                    max_value=max_val
                )
    
    # Make predictions
    if st.button("🎯 Generate Predictions"):
        model = st.session_state.trained_models[selected_model]
        
        with st.spinner("Generating predictions..."):
            if input_method == "Use Latest Data Values":
                # Use last row of data as input
                latest_data = st.session_state.cleaned_data[st.session_state.feature_columns].iloc[-1:]
                input_features = latest_data
            else:
                # Use manual input
                input_features = pd.DataFrame([input_values])
            
            # Generate predictions for multiple days
            predictions = []
            confidence_intervals = []
            
            for day in range(prediction_days):
                pred = model.predict(input_features)[0]
                predictions.append(pred)
                
                # Simple confidence interval calculation
                # In practice, you might want to use more sophisticated methods
                residual_std = st.session_state.model_metrics[selected_model]['RMSE']
                ci_lower = pred - 1.96 * residual_std
                ci_upper = pred + 1.96 * residual_std
                confidence_intervals.append((ci_lower, ci_upper))
            
            # Create prediction dataframe
            future_dates = [
                datetime.now() + timedelta(days=i+1) 
                for i in range(prediction_days)
            ]
            
            predictions_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted_Value': predictions,
                'CI_Lower': [ci[0] for ci in confidence_intervals],
                'CI_Upper': [ci[1] for ci in confidence_intervals]
            })
            
            st.session_state.predictions = predictions_df
        
        # Display predictions
        st.subheader(f"📈 {st.session_state.target_variable} Predictions")
        
        # Predictions table
        st.dataframe(predictions_df, use_container_width=True)
        
        # Predictions chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=predictions_df['Date'],
            y=predictions_df['Predicted_Value'],
            mode='lines+markers',
            name='Predicted Values',
            line=dict(color='blue', width=3)
        ))
        
        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=list(predictions_df['Date']) + list(predictions_df['Date'][::-1]),
            y=list(predictions_df['CI_Upper']) + list(predictions_df['CI_Lower'][::-1]),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='95% Confidence Interval'
        ))
        
        fig.update_layout(
            title=f"{st.session_state.target_variable} Prediction for Next {prediction_days} Days",
            xaxis_title="Date",
            yaxis_title=st.session_state.target_variable,
            hovermode='x'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Download predictions
        st.subheader("📥 Export Predictions")
        csv = predictions_df.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name=f"weather_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
