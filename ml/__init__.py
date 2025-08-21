"""
Machine Learning module for TradeML.

This module provides tools for training and evaluating ML models on trading data.
"""

from .models import BaseModel, RandomForestModel
from .feature_engineering import FeatureEngineer
from .feature_importance import FeatureImportance, FeatureImportanceAnalyzer
from .multiple_samples import MultipleSamplesDataset, MultipleSamplesTrainer, MultipleSamplesEvaluator
from .performance_integration import MLPerformanceIntegrator

__version__ = "0.1.0"

__all__ = [
    'BaseModel',
    'RandomForestModel', 
    'FeatureEngineer',
    'FeatureImportance',
    'FeatureImportanceAnalyzer',
    'MultipleSamplesDataset',
    'MultipleSamplesTrainer',
    'MultipleSamplesEvaluator',
    'MLPerformanceIntegrator'
] 