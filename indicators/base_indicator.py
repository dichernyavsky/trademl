from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseIndicator(ABC):
    """
    Base class for all technical indicators.
    """
    
    def __init__(self):
        """
        Initialize the indicator.
        
        Args:
            name (str, optional): Name of the indicator
        """
        self.is_calculated = False
        self.values = None
        # Don't call get_column_names here as it's abstract
        # self.column_names = self.get_column_names()
        

    def _calculate_for_single_df(self, df, append=True, **kwargs):
        """
        Calculate the indicator values for a single dataframe.
        
        Args:
            df (pd.DataFrame): Input data
            append (bool): Whether to append to original DataFrame
            **kwargs: Additional parameters
            
        Returns:
            pd.DataFrame: Data with indicator values
        """
        pass
    
    def calculate(self, data, append=True, **kwargs):
        """
        Calculate indicator for different data structures.
        
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
    
    @abstractmethod
    def get_column_names(self, **kwargs):
        """
        Get the column names that this indicator produces.
        Override in subclasses to return actual column names.
        
        Returns:
            list: List of column names
        """
        pass
    
    def _append_to_df(self, data, indicator_data):
        """
        Append indicator values to the original DataFrame.
        """
        result = data.copy()
        
        column_names = self.get_column_names()
        for column_name in column_names:
            result[column_name] = indicator_data[column_name]
                
        return result
    
    def _create_indicator_df(self, data, indicator_data):
        """
        Create a new DataFrame with only indicator values.
        """
        result = pd.DataFrame(index=data.index)
        
        column_names = self.get_column_names()
        for column_name in column_names:
            result[column_name] = indicator_data[column_name]
                
        return result