from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class BaseIndicator(ABC):
    """
    Base class for all technical indicators.
    """
    
    def __init__(self, name=None):
        """
        Initialize the indicator.
        
        Args:
            name (str, optional): Name of the indicator
        """
        self.name = name or self.__class__.__name__
        self.is_calculated = False
        self.values = None
        
    @abstractmethod
    def calculate(self, data, append=True, **kwargs):
        """
        Calculate the indicator values.
        
        Args:
            data (pd.DataFrame): Input data
            append (bool): Whether to append to original DataFrame
            **kwargs: Additional parameters
            
        Returns:
            pd.DataFrame: Data with indicator values
        """
        pass
    
    def calculate_for_dict(self, data_dict, append=True, **kwargs):
        """
        Calculate indicator for a dictionary of dataframes.
        
        Args:
            data_dict (dict): Dictionary of DataFrames {symbol: df}
            append (bool): Whether to append to original DataFrames
            **kwargs: Additional parameters
            
        Returns:
            dict: Dictionary of DataFrames with indicators
        """
        result = {}
        for symbol, df in data_dict.items():
            result[symbol] = self.calculate(df, append=append, **kwargs)
        return result
    
    def get_column_names(self, **kwargs):
        """
        Get the column names that this indicator produces.
        Override in subclasses to return actual column names.
        
        Returns:
            list: List of column names
        """
        return [self.name]
    
    def _append_to_df(self, data, indicator_data):
        """
        Append indicator values to the original DataFrame.
        """
        result = data.copy()
        
        if isinstance(indicator_data, dict):
            for key, values in indicator_data.items():
                if isinstance(values, np.ndarray) and len(values) == len(result):
                    result[key] = values
                elif isinstance(values, pd.Series) and len(values) == len(result):
                    result[key] = values
        else:
            column_name = self.name
            if len(indicator_data) == len(result):
                result[column_name] = indicator_data
                
        return result
    
    def _create_indicator_df(self, data, indicator_data):
        """
        Create a new DataFrame with only indicator values.
        """
        result = pd.DataFrame(index=data.index)
        
        if isinstance(indicator_data, dict):
            for key, values in indicator_data.items():
                if isinstance(values, np.ndarray) and len(values) == len(result):
                    result[key] = values
                elif isinstance(values, pd.Series) and len(values) == len(result):
                    result[key] = values
        else:
            column_name = self.name
            if len(indicator_data) == len(result):
                result[column_name] = indicator_data
                
        return result