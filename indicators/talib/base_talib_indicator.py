"""
Base class for TA-Lib indicators with column name mapping

This module provides a base class that handles the mapping between
different column naming conventions and TA-Lib requirements.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

from ..base_indicator import BaseIndicator


class BaseTALibIndicator(BaseIndicator):
    """
    Base class for TA-Lib indicators with automatic column name mapping.
    
    This class handles the conversion between different column naming conventions
    (e.g., 'Close' vs 'close', 'High' vs 'high') and TA-Lib requirements.
    """
    
    def __init__(self):
        super().__init__()
        # Define column name mappings
        self.column_mappings = {
            # Standard TA-Lib names -> possible variations
            'open': ['open', 'Open', 'OPEN'],
            'high': ['high', 'High', 'HIGH'],
            'low': ['low', 'Low', 'LOW'],
            'close': ['close', 'Close', 'CLOSE'],
            'volume': ['volume', 'Volume', 'VOLUME', 'vol', 'Vol', 'VOL']
        }
    
    def _map_columns(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Map column names to TA-Lib standard names and cast to float64.
        
        Args:
            data: DataFrame with potentially non-standard column names
            
        Returns:
            Tuple of (mapped_dataframe, reverse_mapping)
        """
        mapped_data = data.copy()
        reverse_mapping = {}
        
        # Find and map columns
        for talib_name, possible_names in self.column_mappings.items():
            for possible_name in possible_names:
                if possible_name in data.columns:
                    if possible_name != talib_name:
                        mapped_data[talib_name] = data[possible_name]
                        reverse_mapping[talib_name] = possible_name
                    break
        # Ensure all TA-Lib columns are float64
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in mapped_data.columns:
                mapped_data[col] = mapped_data[col].astype(np.float64)
        
        return mapped_data, reverse_mapping
    
    def _restore_columns(self, data: pd.DataFrame, reverse_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Restore original column names after TA-Lib calculation.
        
        Args:
            data: DataFrame with TA-Lib standard column names
            reverse_mapping: Mapping from TA-Lib names to original names
            
        Returns:
            DataFrame with original column names restored
        """
        result = data.copy()
        
        # Remove temporary TA-Lib columns
        for talib_name in reverse_mapping:
            if talib_name in result.columns:
                del result[talib_name]
        
        return result
    
    def _calculate_for_single_df(self, data: pd.DataFrame, append: bool = True, **kwargs):
        """
        Calculate indicator with automatic column mapping.
        
        Args:
            data: DataFrame with OHLCV data
            append: Whether to append to original DataFrame
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with indicator values added
        """
        # Map columns to TA-Lib standard names
        mapped_data, reverse_mapping = self._map_columns(data)
        
        # Calculate indicator using mapped data
        indicator_data = self._calculate_talib(mapped_data, **kwargs)
        
        # Restore original column names
        result = self._restore_columns(mapped_data, reverse_mapping)
        
        # Store values for consistency with BaseIndicator
        self.values = indicator_data
        self.is_calculated = True
        
        # Add indicator columns to result
        if append:
            for col_name, values in indicator_data.items():
                result[col_name] = values
            return result
        else:
            # Create new DataFrame with only indicator values
            indicator_df = pd.DataFrame(index=data.index)
            for col_name, values in indicator_data.items():
                indicator_df[col_name] = values
            return indicator_df
    
    def calculate(self, data, append=True, **kwargs):
        """
        Calculate indicator for different data structures with column mapping.
        
        Args:
            data: Can be:
                - pd.DataFrame: Single DataFrame
                - dict: Dictionary of DataFrames {symbol: df} (simplified format)
                - dict: Dictionary of dictionaries {interval: {symbol: df}} (legacy format)
            append (bool): Whether to append to original DataFrames
            **kwargs: Additional parameters
            
        Returns:
            Same structure as input data with indicators added
        """
        if isinstance(data, pd.DataFrame):
            return self._calculate_for_single_df(data, append=append, **kwargs)
        elif isinstance(data, dict):
            # Check if this is the legacy format with intervals
            if any(isinstance(v, dict) for v in data.values()):
                # Legacy format: {interval: {symbol: df}}
                result = {}
                for interval, symbols_data in data.items():
                    result[interval] = {}
                    for symbol, df in symbols_data.items():
                        result[interval][symbol] = self._calculate_for_single_df(df, append=append, **kwargs)
                return result
            else:
                # Simplified format: {symbol: df} - OPTIMIZED
                # Create a copy of the data dict to avoid modifying the original
                result = {}
                for symbol, df in data.items():
                    result[symbol] = self._calculate_for_single_df(df, append=append, **kwargs)
                return result
        else:
            raise ValueError(f"Invalid data type: {type(data)}")
    
    def _calculate_talib(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """
        Calculate TA-Lib indicator values.
        
        This method should be implemented by subclasses to perform
        the actual TA-Lib calculation.
        
        Args:
            data: DataFrame with TA-Lib standard column names
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping column names to indicator values
        """
        raise NotImplementedError("Subclasses must implement _calculate_talib")
    
    def _validate_required_columns(self, data: pd.DataFrame, required_columns: List[str]):
        """
        Validate that required columns are present in the data.
        
        Args:
            data: DataFrame to validate
            required_columns: List of required column names
            
        Raises:
            ValueError: If required columns are missing
        """
        missing_columns = []
        for col in required_columns:
            if col not in data.columns:
                missing_columns.append(col)
        
        if missing_columns:
            raise ValueError(f"DataFrame must contain columns: {missing_columns}") 