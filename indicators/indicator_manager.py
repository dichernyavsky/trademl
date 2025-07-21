import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from typing import Dict, Union
from .higher_timeframe import HigherTimeframeIndicator
from .base_indicator import BaseIndicator

class IndicatorManager:
    """
    Manager class for handling multiple indicators and their column tracking.
    """
    
    def __init__(self, use_parallel=False, max_workers=None):
        self.indicators = []
        self.indicator_columns = []
        self.use_parallel = use_parallel
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 4)
    
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
        # Create a single copy of the data dict
        result = {symbol: df.copy() for symbol, df in data_dict.items()}
        
        if self.use_parallel and len(self.indicators) > 1:
            return self._calculate_parallel(result, append)
        else:
            # Process all indicators sequentially
            for indicator, kwargs in self.indicators:
                result = indicator.calculate(result, append=True, **kwargs)
            return result
    
    def _calculate_parallel(self, data_dict, append):
        """Calculate indicators in parallel for better performance."""
        def calculate_indicator(indicator_data):
            indicator, kwargs = indicator_data
            return indicator.calculate(data_dict, append=True, **kwargs)
        
        # Process indicators in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(calculate_indicator, (indicator, kwargs)) 
                      for indicator, kwargs in self.indicators]
            
            # Collect results and merge them
            for future in as_completed(futures):
                result = future.result()
                # Merge the results
                for symbol in result:
                    if symbol in data_dict:
                        for col in result[symbol].columns:
                            if col not in data_dict[symbol].columns:
                                data_dict[symbol][col] = result[symbol][col]
        
        return data_dict
    
    def _calculate_for_single(self, data, append):
        """Calculate indicators for single dataframe."""
        result = data.copy()
        
        for indicator, kwargs in self.indicators:
            result = indicator.calculate(result, append=True, **kwargs)
            
        return result
    
    def get_indicator_columns(self):
        """Get all indicator column names."""
        return self.indicator_columns.copy()


class HigherTimeframeIndicatorManager:
    """
    Manager for handling multiple higher timeframe indicators.
    
    This class provides a convenient interface for adding and calculating
    multiple higher timeframe indicators at once.
    """
    
    def __init__(self):
        self.htf_indicators = []
        self.htf_indicator_columns = []
    
    def add_higher_timeframe_indicator(self, source_indicator: BaseIndicator,
                                     higher_timeframe_interval: str = '1h',
                                     lower_timeframe_interval: str = '1m',
                                     alignment_method: str = 'forward_fill',
                                     shift_by_one_period: bool = True):
        """
        Add a higher timeframe indicator to the manager.
        
        Args:
            source_indicator: Indicator to calculate on higher timeframe
            higher_timeframe_interval: Interval of higher timeframe (e.g., '1h', '4h', '1d')
            lower_timeframe_interval: Interval of lower timeframe (e.g., '1m', '5m', '1h')
            alignment_method: Method for aligning values
            shift_by_one_period: Whether to shift indicator values by one period to avoid look-ahead bias
        """
        htf_indicator = HigherTimeframeIndicator(
            source_indicator=source_indicator,
            higher_timeframe_interval=higher_timeframe_interval,
            lower_timeframe_interval=lower_timeframe_interval,
            alignment_method=alignment_method,
            shift_by_one_period=shift_by_one_period
        )
        
        self.htf_indicators.append(htf_indicator)
        
        # Track column names
        column_names = htf_indicator.get_column_names()
        self.htf_indicator_columns.extend(column_names)
    
    def calculate_all(self, lower_timeframe_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                     higher_timeframe_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                     append: bool = True):
        """
        Calculate all higher timeframe indicators for data.
        
        Args:
            lower_timeframe_data: Lower timeframe data (DataFrame or dict of DataFrames)
            higher_timeframe_data: Higher timeframe data (DataFrame or dict of DataFrames)
            append: Whether to append to original data
            
        Returns:
            Enriched data with all higher timeframe indicators
        """
        if isinstance(lower_timeframe_data, dict):
            return self._calculate_for_dict(lower_timeframe_data, higher_timeframe_data, append)
        else:
            return self._calculate_for_single(lower_timeframe_data, higher_timeframe_data, append)
    
    def _calculate_for_dict(self, data_dict: Dict[str, pd.DataFrame], 
                           higher_timeframe_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                           append: bool):
        """Calculate indicators for dict of dataframes."""
        # Start with a copy of the original data
        result = {symbol: df.copy() for symbol, df in data_dict.items()}
        
        # Process each indicator sequentially, updating the result
        for indicator in self.htf_indicators:
            # Calculate this indicator and update the result
            indicator_result = indicator.calculate(result, higher_timeframe_data, append=True)
            result = indicator_result
            
        return result
    
    def _calculate_for_single(self, data: pd.DataFrame, 
                             higher_timeframe_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                             append: bool):
        """Calculate indicators for single dataframe."""
        result = data.copy()
        
        for indicator in self.htf_indicators:
            result = indicator.calculate(result, higher_timeframe_data, append=True)
            
        return result
    
    def get_indicator_columns(self):
        """Get all higher timeframe indicator column names."""
        return self.htf_indicator_columns.copy()
    
    def get_htf_columns(self):
        """Get all higher timeframe indicator column names (alias for compatibility)."""
        return self.htf_indicator_columns.copy()