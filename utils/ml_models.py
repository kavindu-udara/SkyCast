import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st
import joblib
import os

class WeatherMLModels:
    """
    A class to handle machine learning model training and evaluation
    for weather prediction tasks.
    """
    
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        }
    
    def train_model(self, X, y, model_name, test_size=0.2, random_state=42):
        """
        Train a machine learning model and evaluate its performance.
        
        Args:
            X (pd.DataFrame): Feature data
            y (pd.Series): Target data
            model_name (str): Name of the model to train
            test_size (float): Proportion of dataset for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: Trained model and performance metrics
        """
        try:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Get the model
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not supported")
            
            model = self.models[model_name]
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
            
            # Add model-specific metrics
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X.columns.tolist(),
                    'importance': model.feature_importances_.tolist()
                }).sort_values('importance', ascending=False)
                # Convert to dictionary to avoid JSON serialization issues
                metrics['feature_importance'] = {
                    'features': feature_importance['feature'].tolist(),
                    'importances': feature_importance['importance'].tolist()
                }
            
            return model, metrics
            
        except Exception as e:
            st.error(f"Error training {model_name}: {str(e)}")
            return None, None
    
    def _calculate_metrics(self, y_train, y_pred_train, y_test, y_pred_test):
        """
        Calculate performance metrics for regression models.
        
        Args:
            y_train (pd.Series): Training target values
            y_pred_train (np.array): Training predictions
            y_test (pd.Series): Test target values
            y_pred_test (np.array): Test predictions
            
        Returns:
            dict: Performance metrics
        """
        metrics = {
            'Train_MAE': mean_absolute_error(y_train, y_pred_train),
            'Test_MAE': mean_absolute_error(y_test, y_pred_test),
            'Train_RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'Train_R2': r2_score(y_train, y_pred_train),
            'Test_R2': r2_score(y_test, y_pred_test),
            'MAE': mean_absolute_error(y_test, y_pred_test),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'R²': r2_score(y_test, y_pred_test)
        }
        
        return metrics
    
    def save_model(self, model, model_name, target_variable):
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model object
            model_name (str): Name of the model
            target_variable (str): Target variable name
        """
        try:
            os.makedirs("models", exist_ok=True)
            filename = f"models/{model_name.lower().replace(' ', '_')}_{target_variable}.joblib"
            joblib.dump(model, filename)
            st.success(f"Model saved as {filename}")
        except Exception as e:
            st.error(f"Error saving model: {str(e)}")
    
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the model file
            
        Returns:
            Trained model object
        """
        try:
            model = joblib.load(filepath)
            st.success(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None
    
    def compare_models(self, models_metrics):
        """
        Compare performance of multiple models.
        
        Args:
            models_metrics (dict): Dictionary of model metrics
            
        Returns:
            pd.DataFrame: Comparison dataframe
        """
        comparison_data = []
        
        for model_name, metrics in models_metrics.items():
            comparison_data.append({
                'Model': model_name,
                'Test_MAE': metrics['Test_MAE'],
                'Test_RMSE': metrics['Test_RMSE'],
                'Test_R²': metrics['Test_R2']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Rank models (lower MAE and RMSE, higher R² is better)
        comparison_df['MAE_Rank'] = comparison_df['Test_MAE'].rank(ascending=True)
        comparison_df['RMSE_Rank'] = comparison_df['Test_RMSE'].rank(ascending=True)
        comparison_df['R²_Rank'] = comparison_df['Test_R²'].rank(ascending=False)
        comparison_df['Overall_Rank'] = (
            comparison_df['MAE_Rank'] + 
            comparison_df['RMSE_Rank'] + 
            comparison_df['R²_Rank']
        ) / 3
        
        return comparison_df.sort_values('Overall_Rank')
    
    def get_model_recommendations(self, comparison_df):
        """
        Get model recommendations based on performance.
        
        Args:
            comparison_df (pd.DataFrame): Model comparison dataframe
            
        Returns:
            dict: Recommendations
        """
        best_model = comparison_df.iloc[0]['Model']
        best_r2 = comparison_df.loc[comparison_df['Test_R²'].idxmax(), 'Model']
        best_mae = comparison_df.loc[comparison_df['Test_MAE'].idxmin(), 'Model']
        
        recommendations = {
            'overall_best': best_model,
            'best_accuracy': best_r2,
            'best_precision': best_mae,
            'summary': f"Recommended model: {best_model}"
        }
        
        return recommendations
    
    def predict_with_confidence(self, model, X, n_bootstrap=100):
        """
        Make predictions with confidence intervals using bootstrap method.
        
        Args:
            model: Trained model
            X (pd.DataFrame): Features for prediction
            n_bootstrap (int): Number of bootstrap samples
            
        Returns:
            dict: Predictions with confidence intervals
        """
        try:
            # Basic prediction
            base_prediction = model.predict(X)
            
            # For tree-based models, use built-in uncertainty estimation
            if hasattr(model, 'estimators_'):
                # Get predictions from all trees
                tree_predictions = np.array([
                    tree.predict(X) for tree in model.estimators_
                ]).T
                
                # Calculate statistics
                predictions_mean = np.mean(tree_predictions, axis=1)
                predictions_std = np.std(tree_predictions, axis=1)
                
                # 95% confidence interval
                ci_lower = predictions_mean - 1.96 * predictions_std
                ci_upper = predictions_mean + 1.96 * predictions_std
                
            else:
                # For other models, use simple standard deviation estimate
                predictions_mean = base_prediction
                predictions_std = np.full_like(base_prediction, np.std(base_prediction) * 0.1)
                ci_lower = predictions_mean - 1.96 * predictions_std
                ci_upper = predictions_mean + 1.96 * predictions_std
            
            return {
                'predictions': predictions_mean,
                'confidence_lower': ci_lower,
                'confidence_upper': ci_upper,
                'std': predictions_std
            }
            
        except Exception as e:
            st.error(f"Error calculating confidence intervals: {str(e)}")
            return {
                'predictions': model.predict(X),
                'confidence_lower': None,
                'confidence_upper': None,
                'std': None
            }
