"""
Random Forest model for trading ML.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .base_model import BaseModel


class RandomForestModel(BaseModel):
    """
    Random Forest model for trading classification/regression.
    """
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, random_state=42, **kwargs):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required at leaf node
            random_state: Random seed
            **kwargs: Additional parameters
        """
        super().__init__(model_type='classification', **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        
    def _create_model(self, **kwargs):
        """
        Create Random Forest model.
        
        Args:
            **kwargs: Model parameters
            
        Returns:
            RandomForestClassifier or RandomForestRegressor
        """
        if self.model_type == 'classification':
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                **kwargs
            )
        else:
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                **kwargs
            ) 