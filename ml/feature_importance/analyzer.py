"""
Feature Importance Analyzer for trading strategies.

This module provides a high-level interface for analyzing feature importance
in trading ML models, specifically designed to work with our trading data format.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Union
from .core import FeatureImportance


class FeatureImportanceAnalyzer:
    """
    High-level analyzer for feature importance in trading strategies.
    
    This class provides convenient methods for analyzing feature importance
    on trading data with automatic feature detection and filtering.
    """
    
    def __init__(self, 
                 base_estimator: Union[str, object] = 'rf',
                 methods: List[str] = None,
                 cv_splits: int = 5,
                 embargo_pct: float = 0.01,
                 **kwargs):
        """
        Initialize FeatureImportanceAnalyzer.
        
        Args:
            base_estimator: Base estimator for importance calculation
            methods: List of importance methods ('mdi', 'mda', 'sfi', 'ofi')
            cv_splits: Number of CV splits
            embargo_pct: Embargo percentage for time series CV
            **kwargs: Additional arguments for FeatureImportance
        """
        self.fi = FeatureImportance(
            base_estimator=base_estimator,
            methods=methods or ['mdi', 'mda'],
            cv_splits=cv_splits,
            embargo_pct=embargo_pct,
            **kwargs
        )
        self.importance_df = None
        self.feature_columns = None
        
    def analyze_trades(self, 
                      trades_df: pd.DataFrame, 
                      feature_columns: Optional[List[str]] = None,
                      target_column: str = 'bin',
                      weight_column: Optional[str] = 'w',
                      **kwargs) -> pd.DataFrame:
        """
        Analyze feature importance on trading data.
        
        Args:
            trades_df: DataFrame with trades and features
            feature_columns: List of feature columns (auto-detected if None)
            target_column: Target variable column
            weight_column: Sample weight column
            **kwargs: Additional arguments for fit_compute
            
        Returns:
            DataFrame with feature importance scores
        """
        # Auto-detect feature columns if not provided
        if feature_columns is None:
            feature_columns = self._detect_feature_columns(trades_df, target_column, weight_column)
        
        self.feature_columns = feature_columns
        
        # Prepare data
        X = trades_df[feature_columns]
        y = trades_df[target_column]
        sample_weight = trades_df.get(weight_column, None) if weight_column else None
        
        # Compute importance
        self.importance_df = self.fi.fit_compute(X, y, sample_weight=sample_weight, **kwargs)
        
        return self.importance_df
    
    def _detect_feature_columns(self, 
                               trades_df: pd.DataFrame,
                               target_column: str = 'bin',
                               weight_column: Optional[str] = 'w') -> List[str]:
        """
        Automatically detect feature columns in trades DataFrame.
        
        Args:
            trades_df: DataFrame with trades
            target_column: Target variable column
            weight_column: Sample weight column
            
        Returns:
            List of feature column names
        """
        # Standard columns to exclude
        exclude_columns = {
            'Open', 'High', 'Low', 'Close', 'Volume',
            'entry_time', 'exit_time', 'entry_price', 'exit_price',
            'direction', 'pnl', 'return', 'duration'
        }
        
        # Add target and weight columns to exclude
        exclude_columns.add(target_column)
        if weight_column:
            exclude_columns.add(weight_column)
        
        # Filter columns
        feature_columns = []
        for col in trades_df.columns:
            if col not in exclude_columns:
                # Check if column contains numeric data
                if pd.api.types.is_numeric_dtype(trades_df[col]):
                    feature_columns.append(col)
        
        return feature_columns
    
    def filter_features(self, 
                       trades_df: pd.DataFrame,
                       threshold: float = 0.01,
                       method: str = 'mean',
                       **kwargs) -> List[str]:
        """
        Filter features by importance threshold.
        
        Args:
            trades_df: DataFrame with trades
            threshold: Importance threshold
            method: Method to use for filtering ('mean' or specific method)
            **kwargs: Additional arguments for analyze_trades
            
        Returns:
            List of important feature names
        """
        # Analyze if not already done
        if self.importance_df is None:
            self.analyze_trades(trades_df, **kwargs)
        
        # Get importance scores
        if method == 'mean':
            importance_scores = self.importance_df.mean(axis=1)
        else:
            importance_scores = self.importance_df[method]
        
        # Filter by threshold
        important_features = importance_scores[importance_scores > threshold].index.tolist()
        
        return important_features
    
    def get_top_features(self, n: int = 10, method: str = 'mean') -> pd.Series:
        """
        Get top N features by importance.
        
        Args:
            n: Number of top features
            method: Method to use ('mean' or specific method name)
            
        Returns:
            Series with top features
        """
        if self.importance_df is None:
            raise ValueError("Must call analyze_trades first")
        
        return self.fi.get_top_features(n, method)
    
    def plot_importance(self, n: int = 20, method: str = 'mean', **kwargs):
        """
        Plot feature importance.
        
        Args:
            n: Number of features to plot
            method: Method to use for plotting
            **kwargs: Additional plotting arguments
        """
        if self.importance_df is None:
            raise ValueError("Must call analyze_trades first")
        
        self.fi.plot_importance(n, method, **kwargs)
    
    def compare_methods(self, n: int = 10) -> pd.DataFrame:
        """
        Compare feature importance across different methods.
        
        Args:
            n: Number of top features to compare
            
        Returns:
            DataFrame with comparison
        """
        if self.importance_df is None:
            raise ValueError("Must call analyze_trades first")
        
        # Get top features by mean importance
        top_features = self.get_top_features(n, 'mean').index
        
        # Create comparison DataFrame
        comparison_df = self.importance_df.loc[top_features]
        
        return comparison_df
    
    def save_results(self, filepath: str, format: str = 'csv'):
        """
        Save feature importance results to file.
        
        Args:
            filepath: Path to save file
            format: File format ('csv', 'excel', 'json')
        """
        if self.importance_df is None:
            raise ValueError("No results to save. Call analyze_trades first.")
        
        if format == 'csv':
            self.importance_df.to_csv(filepath)
        elif format == 'excel':
            self.importance_df.to_excel(filepath)
        elif format == 'json':
            self.importance_df.to_json(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def load_results(self, filepath: str, format: str = 'csv'):
        """
        Load feature importance results from file.
        
        Args:
            filepath: Path to load file from
            format: File format ('csv', 'excel', 'json')
        """
        if format == 'csv':
            self.importance_df = pd.read_csv(filepath, index_col=0)
        elif format == 'excel':
            self.importance_df = pd.read_excel(filepath, index_col=0)
        elif format == 'json':
            self.importance_df = pd.read_json(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Update feature names
        self.fi.feature_names = self.importance_df.index.tolist()
        self.fi.importance_df = self.importance_df 