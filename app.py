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

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Custom header styling */
    .custom-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .custom-header h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .custom-header p {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Modern sidebar styling */
    .css-1d391kg {
        background: linear-gradient(145deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Card-like containers */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.06);
        margin: 1rem 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
    /* Modern buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }
    
    /* Modern selectbox and inputs */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #E5E7EB;
        transition: border-color 0.2s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #3B82F6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
        border-radius: 10px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 12px 24px;
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        color: white;
    }
    
    /* Success/warning/error messages */
    .stSuccess {
        background: linear-gradient(90deg, #10B981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 8px;
    }
    
    .stWarning {
        background: linear-gradient(90deg, #F59E0B 0%, #D97706 100%);
        color: white;
        border: none;
        border-radius: 8px;
    }
    
    .stError {
        background: linear-gradient(90deg, #EF4444 0%, #DC2626 100%);
        color: white;
        border: none;
        border-radius: 8px;
    }
    
    /* Data frame styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid rgba(0,0,0,0.06);
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    
    /* Section headers */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #1F2937;
        border-bottom: 3px solid #3B82F6;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* File uploader */
    .stFileUploader > div {
        border: 2px dashed #3B82F6;
        border-radius: 12px;
        background: rgba(59, 130, 246, 0.02);
        padding: 2rem;
        transition: all 0.2s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #1D4ED8;
        background: rgba(59, 130, 246, 0.05);
    }
</style>
""", unsafe_allow_html=True)

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
    # Modern header
    st.markdown("""
    <div class="custom-header">
        <h1>🌤️ Weather Prediction Web App</h1>
        <p>Advanced weather forecasting using machine learning algorithms</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Modern sidebar navigation
    st.sidebar.markdown("""
    <h2 style="color: #1F2937; font-family: 'Inter', sans-serif; font-weight: 600; margin-bottom: 1.5rem;">
        📍 Navigation
    </h2>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["📁 Upload Data", "📊 Explore Data", "🤖 Train Model", "🔮 Predict Weather"],
        format_func=lambda x: x.split(" ", 1)[1]
    )
    
    if page == "📁 Upload Data":
        upload_data_page()
    elif page == "📊 Explore Data":
        explore_data_page()
    elif page == "🤖 Train Model":
        train_model_page()
    elif page == "🔮 Predict Weather":
        predict_weather_page()

def upload_data_page():
    st.markdown('<h2 class="section-header">📁 Upload Weather Data</h2>', unsafe_allow_html=True)
    
    # Sample data option
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 🎯 Option 1: Use Sample Dataset")
        st.markdown("Get started quickly with our pre-loaded weather dataset containing temperature, humidity, rainfall, and more.")
        if st.button("🚀 Load Sample Weather Data", key="sample_data"):
            try:
                sample_data = pd.read_csv("data/sample_weather_data.csv")
                # Convert date column to string to avoid Arrow conversion issues
                if 'date' in sample_data.columns:
                    sample_data['date'] = sample_data['date'].astype(str)
                st.session_state.data = sample_data
                st.success("✅ Sample dataset loaded successfully!")
                st.dataframe(sample_data.head(), use_container_width=True)
            except FileNotFoundError:
                st.error("❌ Sample dataset not found. Please upload your own data.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 📤 Option 2: Upload Your Dataset")
        st.markdown("Upload your own CSV file with weather data for custom analysis and predictions.")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a weather dataset with columns like temperature, humidity, rainfall, etc.",
            key="upload_file"
        )
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                # Convert date columns to string to avoid Arrow conversion issues
                date_columns = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
                for col in date_columns:
                    data[col] = data[col].astype(str)
                st.session_state.data = data
                st.success("✅ Dataset uploaded successfully!")
                st.dataframe(data.head(), use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Data cleaning section
    if st.session_state.data is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">🧹 Data Cleaning & Processing</h3>', unsafe_allow_html=True)
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("**Clean your dataset** by handling missing values, removing duplicates, and normalizing data for optimal model performance.")
        
        if st.button("✨ Clean Dataset", key="clean_data"):
            with st.spinner("🔄 Cleaning dataset..."):
                cleaned_data = data_processor.clean_data(st.session_state.data)
                st.session_state.cleaned_data = cleaned_data
                
                # Show cleaning summary
                st.success("✅ Dataset cleaned successfully!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Original Shape", f"{st.session_state.data.shape[0]} × {st.session_state.data.shape[1]}")
                with col2:
                    st.metric("✨ Cleaned Shape", f"{cleaned_data.shape[0]} × {cleaned_data.shape[1]}")
                with col3:
                    missing_values = st.session_state.data.isnull().sum().sum()
                    st.metric("🚫 Missing Values", missing_values)
                
                # Show data info
                st.markdown("### 📈 Dataset Statistics")
                st.dataframe(cleaned_data.describe(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

def explore_data_page():
    st.markdown('<h2 class="section-header">📊 Explore Weather Data</h2>', unsafe_allow_html=True)
    
    if st.session_state.cleaned_data is None:
        st.warning("⚠️ Please upload and clean your dataset first!")
        return
    
    data = st.session_state.cleaned_data
    
    # Data overview with modern cards
    st.markdown("### 📈 Dataset Overview")
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    
    with col1:
        st.metric("📊 Total Records", f"{len(data):,}")
    with col2:
        st.metric("🔢 Features", len(data.columns))
    with col3:
        st.metric("📅 Data Points", f"{len(data):,}")
    with col4:
        st.metric("❌ Missing Values", data.isnull().sum().sum())
    
    # Interactive data table with modern styling
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">🔍 Interactive Data Table</h3>', unsafe_allow_html=True)
    
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    # Filtering options
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        filter_column = "None"
        if numeric_columns:
            filter_column = st.selectbox("🎯 Filter by column:", ["None"] + numeric_columns)
    with col2:
        filter_range = (0, 1)
        if filter_column != "None":
            min_val = float(data[filter_column].min())
            max_val = float(data[filter_column].max())
            filter_range = st.slider(
                f"📊 Select {filter_column} range:",
                min_val, max_val, (min_val, max_val)
            )
    
    # Apply filters
    filtered_data = data.copy()
    if filter_column != "None":
        filtered_data = filtered_data[
            (filtered_data[filter_column] >= filter_range[0]) & 
            (filtered_data[filter_column] <= filter_range[1])
        ]
    
    # Convert datetime columns to string for display to avoid Arrow serialization issues\n    display_data = filtered_data.copy()\n    for col in display_data.columns:\n        if display_data[col].dtype == 'datetime64[ns]':\n            display_data[col] = display_data[col].dt.strftime('%Y-%m-%d')\n    \n    st.dataframe(display_data, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Visualizations with modern tabs
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">📈 Data Visualizations</h3>', unsafe_allow_html=True)
    
    viz_tab1, viz_tab2, viz_tab3, viz_tab4, viz_tab5 = st.tabs([
        "📈 Time Series", "🔥 Correlation Heatmap", "📊 Distribution", "🗺️ Location Map", "🎉 Fun Insights"
    ])
    
    with viz_tab1:
        visualizations.plot_time_series(filtered_data)
    
    with viz_tab2:
        visualizations.plot_correlation_heatmap(filtered_data)
    
    with viz_tab3:
        visualizations.plot_distributions(filtered_data)
    
    with viz_tab4:
        st.markdown("### 🗺️ Weather Station Locations")
        st.markdown("Explore the geographic distribution of weather stations and visualize weather parameters across different locations.")
        visualizations.plot_weather_locations_map(filtered_data)
    
    with viz_tab5:
        visualizations.show_fun_insights(filtered_data)

def train_model_page():
    st.markdown('<h2 class="section-header">🤖 Train Machine Learning Models</h2>', unsafe_allow_html=True)
    
    if st.session_state.cleaned_data is None:
        st.warning("⚠️ Please upload and clean your dataset first!")
        return
    
    data = st.session_state.cleaned_data
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) < 2:
        st.error("❌ Dataset needs at least 2 numeric columns for training!")
        return
    
    # Model configuration with modern styling
    st.markdown('<h3 class="section-header">⚙️ Model Configuration</h3>', unsafe_allow_html=True)
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    with col1:
        target_variable = st.selectbox(
            "🎯 Select target variable to predict:",
            numeric_columns,
            help="Choose the weather parameter you want to predict"
        )
    
    with col2:
        feature_columns = st.multiselect(
            "🔢 Select feature columns:",
            [col for col in numeric_columns if col != target_variable],
            default=[col for col in numeric_columns if col != target_variable][:5],
            help="Choose the input features for prediction"
        )
    st.markdown("</div>", unsafe_allow_html=True)
    
    if not feature_columns:
        st.error("❌ Please select at least one feature column!")
        return
    
    # Model selection with modern styling
    st.markdown('<h3 class="section-header">📦 Model Selection</h3>', unsafe_allow_html=True)
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    models_to_train = st.multiselect(
        "🎨 Select models to train:",
        ["Linear Regression", "Random Forest", "Gradient Boosting"],
        default=["Random Forest"],
        help="You can train and compare multiple models"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
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
        X = data[feature_columns].copy()
        y = data[target_variable].copy()
        
        # Ensure all features are numeric
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) != len(X.columns):
            non_numeric = [col for col in X.columns if col not in numeric_columns]
            st.warning(f"Excluding non-numeric columns: {non_numeric}")
            X = X[numeric_columns]
        
        # Check if we have any features left
        if X.empty or len(X.columns) == 0:
            st.error("No numeric features available for training!")
            return
            
        # Scale the features
        data_processor = DataProcessor()
        X_scaled = data_processor.normalize_features(X)
        
        # Ensure target is numeric
        if not pd.api.types.is_numeric_dtype(y):
            st.error(f"Target variable '{target_variable}' must be numeric for regression models!")
            return
        
        st.success(f"Using {len(X_scaled.columns)} features for training: {list(X_scaled.columns)}")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        trained_models = {}
        model_metrics = {}
        
        for i, model_name in enumerate(models_to_train):
            status_text.text(f"Training {model_name}...")
            progress_bar.progress((i + 1) / len(models_to_train))
            
            try:
                model, metrics = ml_models.train_model(
                    X_scaled, y, model_name, test_size, random_state
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
    st.markdown('<h2 class="section-header">🔮 Weather Prediction</h2>', unsafe_allow_html=True)
    
    if not st.session_state.trained_models:
        st.warning("⚠️ Please train at least one model first!")
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
    
    input_values = {}
    if input_method == "Manual Input":
        st.subheader("Enter Weather Parameters")
        
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
            
            # Generate realistic time-series predictions
            predictions = []
            confidence_intervals = []
            current_features = input_features.copy()
            
            # Get model's training data statistics for realistic variation
            training_data = st.session_state.cleaned_data[st.session_state.feature_columns]
            residual_std = st.session_state.model_metrics[selected_model]['RMSE']
            
            # Get historical data for trend analysis
            target_data = st.session_state.cleaned_data[st.session_state.target_variable]
            historical_mean = target_data.mean()
            historical_std = target_data.std()
            
            for day in range(prediction_days):
                # Add temporal variation to features
                varied_features = current_features.copy()
                
                # Add some realistic day-to-day variation based on historical patterns
                for feature in current_features.columns:
                    if feature in training_data.columns:
                        feature_std = training_data[feature].std()
                        # Add small random variation (3% of standard deviation per day)
                        daily_variation = np.random.normal(0, feature_std * 0.03)
                        varied_features[feature] = varied_features[feature].iloc[0] + daily_variation
                        
                        # Keep within reasonable bounds
                        feature_min = training_data[feature].min()
                        feature_max = training_data[feature].max()
                        varied_features[feature] = np.clip(varied_features[feature], feature_min, feature_max)
                
                # Make prediction with varied features
                base_pred = model.predict(varied_features)[0]
                
                # Add temporal trend (slight drift over time)
                trend_factor = 1 + (day * 0.001)  # Very small trend
                pred = base_pred * trend_factor
                
                # Add some realistic noise
                noise = np.random.normal(0, residual_std * 0.1)
                pred += noise
                
                predictions.append(pred)
                
                # Calculate dynamic confidence intervals that widen over time
                uncertainty_growth = 1 + (day * 0.05)  # Uncertainty increases over time
                ci_width = 1.96 * residual_std * uncertainty_growth
                ci_lower = pred - ci_width
                ci_upper = pred + ci_width
                confidence_intervals.append((ci_lower, ci_upper))
                
                # Update current features for next iteration (feedback loop)
                # Use the prediction to influence future predictions slightly
                if len(predictions) > 1:
                    # Subtle feedback: new prediction influences next day's features
                    for feature in current_features.columns:
                        if 'temperature' in feature.lower() and st.session_state.target_variable.lower() in ['temperature', 'temp']:
                            # Temperature has some persistence
                            current_features[feature] = current_features[feature] * 0.9 + pred * 0.1
                        elif 'humidity' in feature.lower() and st.session_state.target_variable.lower() in ['humidity']:
                            # Humidity has some persistence  
                            current_features[feature] = current_features[feature] * 0.95 + pred * 0.05
                
                # Set random seed to add controlled variation
                np.random.seed(42 + day)
            
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
