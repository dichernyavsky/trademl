"""
Base class for all ML models in TradeML.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import pickle
import os
from typing import Optional, Dict, Any, List


class BaseModel(ABC):
    """
    Base class for all machine learning models in TradeML.
    
    This class provides a common interface for training and making predictions
    on trading data (DataFrameTrades).
    """
    
    def __init__(self, model_type: str = 'classification', **kwargs):
        """
        Initialize the base model.
        
        Args:
            model_type: Type of model ('classification' or 'regression')
            **kwargs: Additional model-specific parameters
        """
        self.model_type = model_type
        self.model = None
        self.feature_columns = []
        self.target_column = 'bin'
        self.is_fitted = False
        self.model_params = kwargs
        
    @abstractmethod
    def _create_model(self, **kwargs):
        """
        Create the underlying ML model.
        Must be implemented by subclasses.
        
        Args:
            **kwargs: Model-specific parameters
            
        Returns:
            The ML model instance
        """
        pass
    
    def fit(self, trades_df: pd.DataFrame, feature_columns: Optional[List[str]] = None, 
            target_column: str = 'bin', **kwargs) -> 'BaseModel':
        """
        Train the model on trading data.
        
        Args:
            trades_df: DataFrameTrades with features and target
            feature_columns: List of feature column names (if None, auto-detect)
            target_column: Name of the target column
            **kwargs: Additional training parameters
            
        Returns:
            self: Trained model instance
        """
        if trades_df.empty:
            raise ValueError("Trades DataFrame is empty")
            
        # Set target column
        self.target_column = target_column
        
        # Auto-detect feature columns if not provided
        if feature_columns is None:
            self.feature_columns = self._auto_detect_features(trades_df)
        else:
            self.feature_columns = feature_columns
            
        # Validate columns exist
        self._validate_columns(trades_df)
        
        # Prepare training data
        X, y = self._prepare_training_data(trades_df)
        
        # Create and train model
        self.model = self._create_model(**self.model_params)
        self.model.fit(X, y, **kwargs)
        
        self.is_fitted = True
        return self
    
    def predict(self, trades_df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new trading data.
        
        Args:
            trades_df: DataFrameTrades with features
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        # Validate columns exist
        self._validate_columns(trades_df)
        
        # Prepare prediction data
        X = self._prepare_prediction_data(trades_df)
        
        # Make predictions
        return self.model.predict(X)
    
    def predict_proba(self, trades_df: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions on new trading data.
        
        Args:
            trades_df: DataFrameTrades with features
            
        Returns:
            np.ndarray: Probability predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
            
        # Validate columns exist
        self._validate_columns(trades_df)
        
        # Prepare prediction data
        X = self._prepare_prediction_data(trades_df)
        
        # Check if model supports predict_proba
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise NotImplementedError(f"{self.__class__.__name__} does not support predict_proba")
    
    def _auto_detect_features(self, trades_df: pd.DataFrame) -> List[str]:
        """
        Auto-detect feature columns from trades DataFrame.
        
        Args:
            trades_df: DataFrameTrades
            
        Returns:
            List[str]: List of feature column names
        """
        # Exclude known non-feature columns
        exclude_columns = {
            'bin', 'direction', 'entry_price', 'exit_price', 'exit_time',
            'pt', 'sl', 't1', 'dynamic_stop_path'
        }
        
        # Get all numeric columns that are not in exclude list
        feature_columns = []
        for col in trades_df.columns:
            if (col not in exclude_columns and 
                trades_df[col].dtype in ['int64', 'float64', 'float32', 'int32']):
                feature_columns.append(col)
                
        return feature_columns
    
    def _validate_columns(self, trades_df: pd.DataFrame):
        """
        Validate that required columns exist in trades DataFrame.
        
        Args:
            trades_df: DataFrameTrades to validate
            
        Raises:
            ValueError: If required columns are missing
        """
        missing_features = [col for col in self.feature_columns if col not in trades_df.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")
    
    def _prepare_training_data(self, trades_df: pd.DataFrame) -> tuple:
        """
        Prepare training data from trades DataFrame.
        
        Args:
            trades_df: DataFrameTrades
            
        Returns:
            tuple: (X, y) training data
        """
        # Remove rows with missing target values
        valid_mask = trades_df[self.target_column].notna()
        trades_df = trades_df[valid_mask]
        
        # Prepare features
        X = trades_df[self.feature_columns].values
        
        # Prepare target
        y = trades_df[self.target_column].values
        
        return X, y
    
    def _prepare_prediction_data(self, trades_df: pd.DataFrame) -> np.ndarray:
        """
        Prepare prediction data from trades DataFrame.
        
        Args:
            trades_df: DataFrameTrades
            
        Returns:
            np.ndarray: Feature matrix for prediction
        """
        return trades_df[self.feature_columns].values
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores if available.
        
        Returns:
            Dict[str, float]: Feature importance scores or None if not available
        """
        if not self.is_fitted:
            return None
            
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_columns, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            return dict(zip(self.feature_columns, self.model.coef_[0]))
        else:
            return None
    
    def save(self, path: str):
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        model_state = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)
    
    def load(self, path: str) -> 'BaseModel':
        """
        Load the model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            self: Loaded model instance
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        with open(path, 'rb') as f:
            model_state = pickle.load(f)
        
        # Restore model state
        self.model = model_state['model']
        self.feature_columns = model_state['feature_columns']
        self.target_column = model_state['target_column']
        self.model_type = model_state['model_type']
        self.model_params = model_state['model_params']
        self.is_fitted = model_state['is_fitted']
        
        return self
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Dict[str, Any]: Model parameters
        """
        return {
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_params': self.model_params,
            'is_fitted': self.is_fitted
        } 