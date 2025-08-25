import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

class WeatherVisualizations:
    """
    A class to create various visualizations for weather data analysis.
    """
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
    
    def plot_time_series(self, df):
        """
        Create interactive time series plots for weather parameters.
        
        Args:
            df (pd.DataFrame): Weather dataset
        """
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_columns:
                st.warning("No numeric columns found for time series plotting.")
                return
            
            # Check if there's a date column
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            
            if date_columns:
                x_axis = date_columns[0]
                df[x_axis] = pd.to_datetime(df[x_axis])
            else:
                # Create a simple index for x-axis
                x_axis = 'Index'
                df[x_axis] = range(len(df))
            
            # Select parameters to plot
            selected_params = st.multiselect(
                "Select parameters to plot:",
                numeric_columns,
                default=numeric_columns[:3] if len(numeric_columns) >= 3 else numeric_columns
            )
            
            if not selected_params:
                st.info("Please select at least one parameter to plot.")
                return
            
            # Create subplots
            fig = make_subplots(
                rows=len(selected_params), 
                cols=1,
                subplot_titles=selected_params,
                vertical_spacing=0.05
            )
            
            for i, param in enumerate(selected_params, 1):
                fig.add_trace(
                    go.Scatter(
                        x=df[x_axis],
                        y=df[param],
                        mode='lines',
                        name=param,
                        line=dict(color=self.color_palette[i % len(self.color_palette)]),
                        showlegend=False
                    ),
                    row=i, col=1
                )
            
            fig.update_layout(
                height=300 * len(selected_params),
                title_text="Weather Parameters Over Time",
                showlegend=False
            )
            
            fig.update_xaxes(title_text="Time" if date_columns else "Index")
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating time series plot: {str(e)}")
    
    def plot_correlation_heatmap(self, df):
        """
        Create a correlation heatmap for numeric features.
        
        Args:
            df (pd.DataFrame): Weather dataset
        """
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.empty:
                st.warning("No numeric columns found for correlation analysis.")
                return
            
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr()
            
            # Create heatmap using Plotly
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title="Weather Parameters Correlation Heatmap",
                xaxis_title="Features",
                yaxis_title="Features",
                width=600,
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show highest correlations
            st.subheader("Strongest Correlations")
            
            # Get upper triangle of correlation matrix
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find highest correlations
            correlations = []
            for col in upper_triangle.columns:
                for idx in upper_triangle.index:
                    if not pd.isna(upper_triangle.loc[idx, col]):
                        correlations.append({
                            'Feature 1': idx,
                            'Feature 2': col,
                            'Correlation': upper_triangle.loc[idx, col]
                        })
            
            if correlations:
                corr_df = pd.DataFrame(correlations)
                corr_df['Abs_Correlation'] = corr_df['Correlation'].abs()
                corr_df = corr_df.sort_values('Abs_Correlation', ascending=False)
                corr_df = corr_df.drop('Abs_Correlation', axis=1)
                st.dataframe(corr_df.head(10), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating correlation heatmap: {str(e)}")
    
    def plot_distributions(self, df):
        """
        Create distribution plots for numeric features.
        
        Args:
            df (pd.DataFrame): Weather dataset
        """
        try:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_columns:
                st.warning("No numeric columns found for distribution plotting.")
                return
            
            # Select parameter for distribution
            selected_param = st.selectbox(
                "Select parameter for distribution analysis:",
                numeric_columns
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig_hist = px.histogram(
                    df, 
                    x=selected_param,
                    nbins=30,
                    title=f"Distribution of {selected_param}",
                    marginal="box"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Box plot
                fig_box = px.box(
                    df, 
                    y=selected_param,
                    title=f"Box Plot of {selected_param}"
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            # Statistics summary
            st.subheader(f"Statistics for {selected_param}")
            stats = df[selected_param].describe()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{stats['mean']:.2f}")
            with col2:
                st.metric("Std Dev", f"{stats['std']:.2f}")
            with col3:
                st.metric("Min", f"{stats['min']:.2f}")
            with col4:
                st.metric("Max", f"{stats['max']:.2f}")
            
        except Exception as e:
            st.error(f"Error creating distribution plots: {str(e)}")
    
    def show_fun_insights(self, df):
        """
        Display fun insights and statistics about the weather data.
        
        Args:
            df (pd.DataFrame): Weather dataset
        """
        try:
            st.subheader("🎉 Fun Weather Insights")
            
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if not numeric_columns:
                st.warning("No numeric data available for insights.")
                return
            
            # Temperature insights (if available)
            temp_columns = [col for col in numeric_columns if 'temp' in col.lower()]
            if temp_columns:
                temp_col = temp_columns[0]
                hottest_day = df.loc[df[temp_col].idxmax()]
                coldest_day = df.loc[df[temp_col].idxmin()]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"🔥 **Hottest Day**\n\nTemperature: {hottest_day[temp_col]:.1f}°")
                    if 'date' in [col.lower() for col in df.columns]:
                        date_col = [col for col in df.columns if 'date' in col.lower()][0]
                        st.info(f"Date: {hottest_day[date_col]}")
                
                with col2:
                    st.info(f"🧊 **Coldest Day**\n\nTemperature: {coldest_day[temp_col]:.1f}°")
                    if 'date' in [col.lower() for col in df.columns]:
                        date_col = [col for col in df.columns if 'date' in col.lower()][0]
                        st.info(f"Date: {coldest_day[date_col]}")
            
            # Rainfall insights (if available)
            rain_columns = [col for col in numeric_columns if 'rain' in col.lower() or 'precip' in col.lower()]
            if rain_columns:
                rain_col = rain_columns[0]
                total_rainfall = df[rain_col].sum()
                rainiest_day = df.loc[df[rain_col].idxmax()]
                rainy_days = (df[rain_col] > 0).sum()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"🌧️ **Rainfall Stats**\n\nTotal Rainfall: {total_rainfall:.1f}mm")
                    st.success(f"Rainy Days: {rainy_days}")
                
                with col2:
                    st.success(f"☔ **Rainiest Day**\n\nRainfall: {rainiest_day[rain_col]:.1f}mm")
            
            # Wind insights (if available)
            wind_columns = [col for col in numeric_columns if 'wind' in col.lower()]
            if wind_columns:
                wind_col = wind_columns[0]
                windiest_day = df.loc[df[wind_col].idxmax()]
                avg_wind = df[wind_col].mean()
                
                st.warning(f"💨 **Wind Stats**\n\nAverage Wind Speed: {avg_wind:.1f}")
                st.warning(f"Windiest Day: {windiest_day[wind_col]:.1f}")
            
            # General insights
            st.markdown("---")
            st.subheader("📈 Trends and Patterns")
            
            # Create a summary chart
            if len(numeric_columns) >= 2:
                # Select top 4 numeric columns for radar chart
                selected_cols = numeric_columns[:4]
                
                # Normalize data for radar chart
                normalized_data = df[selected_cols].mean()
                normalized_data = (normalized_data - normalized_data.min()) / (normalized_data.max() - normalized_data.min())
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=normalized_data.values,
                    theta=normalized_data.index,
                    fill='toself',
                    name='Average Values (Normalized)'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="Weather Parameters Overview"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Data quality insights
            st.subheader("📊 Data Quality")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Complete Records", len(df.dropna()))
            with col3:
                completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                st.metric("Data Completeness", f"{completeness:.1f}%")
            
        except Exception as e:
            st.error(f"Error generating fun insights: {str(e)}")
    
    def plot_feature_importance(self, feature_importance_data):
        """
        Plot feature importance from tree-based models.
        
        Args:
            feature_importance_data (dict or pd.DataFrame): Feature importance data
        """
        try:
            # Handle both dictionary and DataFrame formats
            if isinstance(feature_importance_data, dict):
                # Convert dictionary format to DataFrame
                features = feature_importance_data.get('features', [])
                importances = feature_importance_data.get('importances', [])
                
                if len(features) != len(importances):
                    st.error("Feature importance data format is invalid")
                    return
                
                feature_importance_df = pd.DataFrame({
                    'feature': features,
                    'importance': importances
                })
            else:
                feature_importance_df = feature_importance_data
            
            # Take top 10 features
            top_features = feature_importance_df.head(10)
            
            fig = px.bar(
                top_features,
                x='importance',
                y='feature',
                orientation='h',
                title="Top 10 Feature Importance",
                labels={'importance': 'Importance Score', 'feature': 'Features'}
            )
            
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error plotting feature importance: {str(e)}")
    
    def plot_prediction_vs_actual(self, y_true, y_pred, title="Predictions vs Actual"):
        """
        Create scatter plot of predictions vs actual values.
        
        Args:
            y_true (array-like): Actual values
            y_pred (array-like): Predicted values
            title (str): Plot title
        """
        try:
            fig = go.Figure()
            
            # Scatter plot
            fig.add_trace(go.Scatter(
                x=y_true,
                y=y_pred,
                mode='markers',
                name='Predictions',
                marker=dict(
                    opacity=0.6,
                    size=5
                )
            ))
            
            # Perfect prediction line
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Actual Values",
                yaxis_title="Predicted Values",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating prediction plot: {str(e)}")
    
    def plot_weather_locations_map(self, df):
        """
        Create an interactive map showing weather station locations with weather data.
        
        Args:
            df (pd.DataFrame): Weather dataset with latitude and longitude columns
        """
        try:
            # Check if latitude and longitude columns exist
            lat_columns = [col for col in df.columns if 'lat' in col.lower()]
            lon_columns = [col for col in df.columns if 'lon' in col.lower()]
            
            if not lat_columns or not lon_columns:
                st.warning("🗺️ No location data (latitude/longitude) found in the dataset.")
                return
            
            lat_col = lat_columns[0]
            lon_col = lon_columns[0]
            
            # Get numeric columns for color coding
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove lat/lon from color options
            color_options = [col for col in numeric_columns if col not in [lat_col, lon_col]]
            
            if not color_options:
                st.warning("No numeric data available for color coding the map.")
                return
            
            # Let user select parameter for color coding
            color_param = st.selectbox(
                "🎨 Select parameter for color coding:",
                color_options,
                index=0 if color_options else None
            )
            
            if color_param:
                # Create the map
                fig = px.scatter_mapbox(
                    df,
                    lat=lat_col,
                    lon=lon_col,
                    color=color_param,
                    size_max=15,
                    hover_data={
                        col: True for col in df.columns 
                        if col not in [lat_col, lon_col] and col in numeric_columns
                    },
                    title=f"Weather Stations - Colored by {color_param}",
                    color_continuous_scale="Viridis"
                )
                
                # Update map layout
                fig.update_layout(
                    mapbox=dict(
                        style="open-street-map",
                        center=dict(
                            lat=df[lat_col].mean(),
                            lon=df[lon_col].mean()
                        ),
                        zoom=10
                    ),
                    height=600,
                    margin=dict(l=0, r=0, t=50, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show location statistics
                st.subheader("📍 Location Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Unique Locations", len(df[[lat_col, lon_col]].drop_duplicates()))
                with col2:
                    st.metric("Center Latitude", f"{df[lat_col].mean():.4f}°")
                with col3:
                    st.metric("Center Longitude", f"{df[lon_col].mean():.4f}°")
                
                # Show location spread
                lat_range = df[lat_col].max() - df[lat_col].min()
                lon_range = df[lon_col].max() - df[lon_col].min()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Latitude Range", f"{lat_range:.4f}°")
                with col2:
                    st.metric("Longitude Range", f"{lon_range:.4f}°")
            
        except Exception as e:
            st.error(f"Error creating weather locations map: {str(e)}")
