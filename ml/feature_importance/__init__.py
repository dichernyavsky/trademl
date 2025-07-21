"""
Feature Importance module for TradeML.

This module provides tools for analyzing feature importance in trading ML models,
including leakage-aware cross-validation and various importance methods.
"""

from .core import FeatureImportance, PurgedKFold, FeatureOrthogonalizer
from .analyzer import FeatureImportanceAnalyzer

__version__ = "0.1.0"

__all__ = [
    'FeatureImportance',
    'PurgedKFold', 
    'FeatureOrthogonalizer',
    'FeatureImportanceAnalyzer'
] 