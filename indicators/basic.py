"""
Basic technical indicators module.

This module contains fundamental technical indicators like moving averages.
"""

import pandas as pd
import numpy as np
from .base_indicator import BaseIndicator


class SimpleMovingAverage(BaseIndicator):
    """
    Simple Moving Average indicator.
    
    Calculates the arithmetic mean of prices over a specified window.
    """
    
    def __init__(self, window: int = 20, column: str = 'close'):
        """
        Initialize Simple Moving Average.
        
        Args:
            window (int): Number of periods for the moving average
            column (str): Column name to calculate SMA on ('close', 'high', 'low', 'open')
        """
        self.window = window
        self.column = column.lower()
        super().__init__()
    
    def get_column_names(self, **kwargs):
        """Get column names for this indicator."""
        return [f'sma_{self.window}']
    
    def _calculate_for_single_df(self, data: pd.DataFrame, append: bool = True, **kwargs):
        """
        Calculate SMA for a single DataFrame.
        
        Args:
            data: DataFrame with OHLCV data
            append: Whether to append to original DataFrame
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with SMA values
        """
        if data.empty:
            return data
        
        # Map column name to actual column
        column_map = {
            'close': 'close',
            'high': 'high', 
            'low': 'low',
            'open': 'open'
        }
        
        price_column = column_map.get(self.column, 'close')
        
        # Calculate SMA
        sma_values = data[price_column].rolling(window=self.window, min_periods=1).mean()
        
        # Create result
        indicator_data = {f'sma_{self.window}': sma_values}
        
        # Store values for consistency with BaseIndicator
        self.values = indicator_data
        self.is_calculated = True
        
        return self._append_to_df(data, indicator_data) if append else self._create_indicator_df(data, indicator_data) 