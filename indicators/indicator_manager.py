import pandas as pd

class IndicatorManager:
    """
    Manager class for handling multiple indicators and their column tracking.
    """
    
    def __init__(self):
        self.indicators = []
        self.indicator_columns = []
    
    def add_indicator(self, indicator, **kwargs):
        """
        Add an indicator to the manager.
        
        Args:
            indicator: Indicator instance
            **kwargs: Parameters for the indicator
        """
        self.indicators.append((indicator, kwargs))
        # Track column names
        column_names = indicator.get_column_names(**kwargs)
        self.indicator_columns.extend(column_names)
    
    def calculate_all(self, data, append=True):
        """
        Calculate all indicators for data.
        
        Args:
            data: DataFrame or dict of DataFrames
            append: Whether to append to original data
            
        Returns:
            Enriched data with all indicators
        """
        if isinstance(data, dict):
            return self._calculate_for_dict(data, append)
        else:
            return self._calculate_for_single(data, append)
    
    def _calculate_for_dict(self, data_dict, append):
        """Calculate indicators for dict of dataframes."""
        result = {symbol: df.copy() for symbol, df in data_dict.items()}
        
        for indicator, kwargs in self.indicators:
            enriched = indicator.calculate_for_dict(result, append=True, **kwargs)
            result = enriched
            
        return result
    
    def _calculate_for_single(self, data, append):
        """Calculate indicators for single dataframe."""
        result = data.copy()
        
        for indicator, kwargs in self.indicators:
            result = indicator.calculate(result, append=True, **kwargs)
            
        return result
    
    def get_indicator_columns(self):
        """Get all indicator column names."""
        return self.indicator_columns.copy()