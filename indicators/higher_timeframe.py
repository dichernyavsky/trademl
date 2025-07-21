"""
Higher Timeframe Indicators Module.

This module provides functionality to calculate indicators on higher timeframes
and align them to lower timeframes (e.g., 1h indicators to 1m data).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional
from .base_indicator import BaseIndicator


class HigherTimeframeIndicator(BaseIndicator):
    """
    Indicator that calculates values on higher timeframes and aligns them to lower timeframes.
    
    This class allows you to:
    1. Calculate indicators on higher timeframes (e.g., 1h, 1d)
    2. Align these values to lower timeframes (e.g., 1m, 5m)
    3. Use any existing indicator as the source
    
    Example:
        # Create lightweight indicator
        sma_1h = SimpleMovingAverage(window=20)
        htf_indicator = HigherTimeframeIndicator(
            source_indicator=sma_1h,
            higher_timeframe_interval='1h',
            lower_timeframe_interval='1m'
        )
        
        # Calculate with specific data
        df_1m_with_sma = htf_indicator.calculate(
            lower_timeframe_data=df_1m,
            higher_timeframe_data=df_1h
        )
    """
    
    def __init__(self, source_indicator: BaseIndicator,
                 higher_timeframe_interval: str = '1h',
                 lower_timeframe_interval: str = '1m',
                 alignment_method: str = 'forward_fill',
                 shift_by_one_period: bool = True):
        """
        Initialize the higher timeframe indicator.
        
        Args:
            source_indicator: Indicator to calculate on higher timeframe
            higher_timeframe_interval: Interval of higher timeframe (e.g., '1h', '4h', '1d')
            lower_timeframe_interval: Interval of lower timeframe (e.g., '1m', '5m', '1h')
            alignment_method: Method for aligning values ('forward_fill', 'backward_fill', 'nearest')
            shift_by_one_period: Whether to shift indicator values by one period to avoid look-ahead bias
        """
        self.source_indicator = source_indicator
        self.higher_timeframe_interval = higher_timeframe_interval
        self.lower_timeframe_interval = lower_timeframe_interval
        self.alignment_method = alignment_method
        self.shift_by_one_period = shift_by_one_period
        
        super().__init__()
    
    def get_column_names(self, **kwargs):
        """
        Get column names for this indicator.
        
        Returns:
            List[str]: Column names with higher timeframe suffix
        """
        source_columns = self.source_indicator.get_column_names(**kwargs)
        return [f"{col}_{self.higher_timeframe_interval}" for col in source_columns]
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """
        Convert timeframe string to minutes.
        
        Args:
            timeframe: Timeframe string (e.g., '1m', '5m', '1h', '4h', '1d')
            
        Returns:
            int: Number of minutes
        """
        timeframe = timeframe.lower()
        
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 24 * 60
        elif timeframe.endswith('w'):
            return int(timeframe[:-1]) * 7 * 24 * 60
        else:
            raise ValueError(f"Unsupported timeframe format: {timeframe}")
    
    def _calculate_shift_periods(self) -> int:
        """
        Calculate how many periods to shift based on timeframe intervals.
        
        For look-ahead bias avoidance, we shift by one higher timeframe period.
        This means we shift by 1 in the higher timeframe data, regardless of the
        ratio between higher and lower timeframes.
        
        Returns:
            int: Number of periods to shift in higher timeframe data (always 0 or 1)
        """
        if not self.shift_by_one_period:
            return 0
            
        # For look-ahead bias avoidance, we always shift by exactly one higher timeframe period
        # This is a design choice to ensure we don't use future information
        # The actual effect on lower timeframe data depends on the ratio of timeframes
        # and is calculated separately in get_shift_info()
        return 1
    
    def calculate(self, lower_timeframe_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                 higher_timeframe_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                 append: bool = True, **kwargs):
        """
        Calculate higher timeframe indicators and align them to lower timeframe data.
        
        Args:
            lower_timeframe_data: Lower timeframe data (DataFrame or dict of DataFrames)
            higher_timeframe_data: Higher timeframe data (DataFrame or dict of DataFrames)
            append: Whether to append to original DataFrame
            **kwargs: Additional parameters
            
        Returns:
            Enriched data with higher timeframe indicators
        """
        if isinstance(lower_timeframe_data, dict):
            return self._calculate_for_dict(lower_timeframe_data, higher_timeframe_data, append, **kwargs)
        else:
            return self._calculate_for_single_df(lower_timeframe_data, higher_timeframe_data, append, **kwargs)
    
    def _calculate_for_single_df(self, data: pd.DataFrame, higher_timeframe_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                                append: bool = True, **kwargs):
        """
        Calculate higher timeframe indicators for a single DataFrame.
        
        Args:
            data: Lower timeframe DataFrame
            higher_timeframe_data: Higher timeframe data
            append: Whether to append to original DataFrame
            **kwargs: Additional parameters
            
        Returns:
            pd.DataFrame: Data with higher timeframe indicators
        """
        if data.empty:
            return data
        
        # Get indicator columns from source indicator
        source_columns = self.source_indicator.get_column_names(**kwargs)
        target_columns = self.get_column_names(**kwargs)
        
        # Handle higher timeframe data
        if isinstance(higher_timeframe_data, pd.DataFrame):
            higher_tf_df = higher_timeframe_data
        elif isinstance(higher_timeframe_data, dict):
            # Check if this is the legacy format with intervals
            if any(isinstance(v, dict) for v in higher_timeframe_data.values()):
                # Legacy format: {interval: {symbol: df}}
                higher_tf_df = {}
                for interval, symbols_data in higher_timeframe_data.items():
                    higher_tf_df[interval] = {}
                    for symbol, df in symbols_data.items():
                        higher_tf_df[interval][symbol] = df  # Don't calculate here
            else:
                # Simplified format: {symbol: df}
                higher_tf_df = {}
                for symbol, df in higher_timeframe_data.items():
                    higher_tf_df[symbol] = df  # Don't calculate here
        else:
            raise ValueError(f"Invalid higher timeframe data type: {type(higher_timeframe_data)}")
        
        # For single DataFrame, assume we're working with the first symbol
        # or use a default approach
        if isinstance(higher_tf_df, pd.DataFrame):
            higher_tf_with_indicators = higher_tf_df
        elif isinstance(higher_tf_df, dict):
            # Try to find matching symbol or use first available
            if len(higher_tf_df) == 1:
                higher_tf_with_indicators = list(higher_tf_df.values())[0]
            else:
                # For multiple symbols, we need to know which symbol this data represents
                # This is a limitation - we'll use the first available
                higher_tf_with_indicators = list(higher_tf_df.values())[0]
        else:
            raise ValueError("No higher timeframe data available")
        
        # Calculate indicators on higher timeframe data (only once!)
        higher_tf_with_indicators = self.source_indicator.calculate(higher_tf_with_indicators, append=True)
        
        # Extract indicator data and apply shift to avoid look-ahead bias if requested
        indicator_data = higher_tf_with_indicators[source_columns].copy()
        
        if self.shift_by_one_period:
            shift_periods = self._calculate_shift_periods()
            for col in source_columns:
                indicator_data[col] = indicator_data[col].shift(shift_periods)
            # Remove rows which now have NaN values
            indicator_data = indicator_data.dropna()
        
        # Use reindex with forward fill to align higher timeframe indicators to lower timeframe
        # This avoids column name collisions that occur with merge_asof
        higher_ind = indicator_data.reindex(data.index, method='ffill')
        
        # Create result DataFrame
        result = data.copy()
        
        # Add aligned indicator values
        for i, source_col in enumerate(source_columns):
            target_col = target_columns[i]
            result[target_col] = higher_ind[source_col].values
        
        # Store values for consistency with BaseIndicator
        indicator_data_dict = {}
        for target_col in target_columns:
            indicator_data_dict[target_col] = result[target_col]
        
        self.values = indicator_data_dict
        self.is_calculated = True
        
        return self._append_to_df(data, indicator_data_dict) if append else self._create_indicator_df(data, indicator_data_dict)
    
    def _calculate_for_dict(self, data_dict: Dict[str, pd.DataFrame], 
                           higher_timeframe_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                           append: bool = True, **kwargs):
        """
        Calculate higher timeframe indicators for dictionary of DataFrames.
        
        Args:
            data_dict: Dictionary of DataFrames {symbol: df}
            higher_timeframe_data: Higher timeframe data
            append: Whether to append to original DataFrames
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with higher timeframe indicators
        """
        result = {}
        
        for symbol, df in data_dict.items():
            # Get corresponding higher timeframe data for this symbol
            if isinstance(higher_timeframe_data, dict):
                if symbol in higher_timeframe_data:
                    # Use specific higher timeframe data for this symbol
                    symbol_higher_tf_data = higher_timeframe_data[symbol]
                else:
                    # Symbol not found in higher timeframe data, skip
                    result[symbol] = df.copy()
                    continue
            else:
                # Single DataFrame case
                symbol_higher_tf_data = higher_timeframe_data
            
            # Calculate for this symbol
            result[symbol] = self._calculate_for_single_df(df, symbol_higher_tf_data, append=append, **kwargs)
        
        return result

    def get_shift_info(self) -> dict:
        """
        Get information about the shift calculation.
        
        Returns:
            dict: Information about shift periods and lost data
        """
        if not self.shift_by_one_period:
            return {
                'shift_periods': 0,
                'lost_lower_periods': 0,
                'higher_minutes': 0,
                'lower_minutes': 0,
                'periods_per_higher': 0
            }
        
        higher_minutes = self._timeframe_to_minutes(self.higher_timeframe_interval)
        lower_minutes = self._timeframe_to_minutes(self.lower_timeframe_interval)
        periods_per_higher = higher_minutes // lower_minutes
        
        return {
            'shift_periods': 1,  # Always shift by 1 higher timeframe period
            'lost_lower_periods': periods_per_higher,  # How many lower periods are lost
            'higher_minutes': higher_minutes,
            'lower_minutes': lower_minutes,
            'periods_per_higher': periods_per_higher
        }