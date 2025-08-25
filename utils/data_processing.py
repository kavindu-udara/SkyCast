import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st

class DataProcessor:
    """
    A class to handle data processing tasks including cleaning, 
    normalization, and validation for weather datasets.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def clean_data(self, df):
        """
        Clean the dataset by handling missing values, removing duplicates,
        and performing basic data validation.
        
        Args:
            df (pd.DataFrame): Raw dataset
            
        Returns:
            pd.DataFrame: Cleaned dataset
        """
        try:
            # Create a copy to avoid modifying the original
            cleaned_df = df.copy()
            
            # Remove completely empty rows and columns
            cleaned_df = cleaned_df.dropna(how='all').dropna(axis=1, how='all')
            
            # Handle date column if present
            date_columns = [col for col in cleaned_df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                for col in date_columns:
                    try:
                        cleaned_df[col] = pd.to_datetime(cleaned_df[col])
                    except:
                        pass
            
            # Identify numeric columns
            numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
            
            # Handle missing values in numeric columns
            for col in numeric_columns:
                if cleaned_df[col].isnull().sum() > 0:
                    # Use forward fill first, then backward fill, then mean
                    cleaned_df[col] = cleaned_df[col].ffill().bfill()
                    if cleaned_df[col].isnull().sum() > 0:
                        cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            
            # Remove duplicate rows
            initial_rows = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            removed_duplicates = initial_rows - len(cleaned_df)
            
            # Handle outliers using IQR method for numeric columns
            for col in numeric_columns:
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them to preserve data
                cleaned_df[col] = np.clip(cleaned_df[col], lower_bound, upper_bound)
            
            # Data validation
            self._validate_weather_data(cleaned_df)
            
            # Display cleaning summary
            st.info(f"""
            **Data Cleaning Summary:**
            - Original rows: {len(df)}
            - Cleaned rows: {len(cleaned_df)}
            - Duplicates removed: {removed_duplicates}
            - Missing values handled: ✅
            - Outliers capped: ✅
            """)
            
            return cleaned_df
            
        except Exception as e:
            st.error(f"Error during data cleaning: {str(e)}")
            return df
    
    def _validate_weather_data(self, df):
        """
        Validate weather data for reasonable ranges.
        
        Args:
            df (pd.DataFrame): Dataset to validate
        """
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Define reasonable ranges for common weather parameters
        weather_ranges = {
            'temperature': (-50, 60),  # Celsius
            'humidity': (0, 100),      # Percentage
            'pressure': (900, 1100),   # hPa
            'wind_speed': (0, 200),    # km/h
            'rainfall': (0, 500),      # mm
            'visibility': (0, 50)      # km
        }
        
        warnings = []
        
        for col in numeric_columns:
            col_lower = col.lower()
            for param, (min_val, max_val) in weather_ranges.items():
                if param in col_lower:
                    out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                    if out_of_range > 0:
                        warnings.append(f"⚠️ {out_of_range} values in '{col}' may be outside normal range ({min_val}-{max_val})")
        
        if warnings:
            st.warning("Data Validation Warnings:\n" + "\n".join(warnings))
    
    def normalize_features(self, X_train, X_test=None):
        """
        Normalize features using StandardScaler.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame, optional): Test features
            
        Returns:
            tuple: Normalized training and test features
        """
        try:
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            
            if X_test is not None:
                X_test_scaled = pd.DataFrame(
                    self.scaler.transform(X_test),
                    columns=X_test.columns,
                    index=X_test.index
                )
                return X_train_scaled, X_test_scaled
            
            return X_train_scaled
            
        except Exception as e:
            st.error(f"Error during feature normalization: {str(e)}")
            return X_train, X_test
    
    def get_feature_info(self, df):
        """
        Get detailed information about dataset features.
        
        Args:
            df (pd.DataFrame): Dataset
            
        Returns:
            dict: Feature information
        """
        info = {
            'total_features': len(df.columns),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns),
            'datetime_features': len(df.select_dtypes(include=['datetime64']).columns),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum()
        }
        
        return info
