"""
Strategy Indicator Manager with Higher Timeframe Support.

This module provides a specialized indicator manager for strategies that can handle
both regular indicators and higher timeframe indicators.
"""

import pandas as pd
from typing import Dict, List, Union, Optional
from .indicator_manager import IndicatorManager
from .higher_timeframe import HigherTimeframeIndicator
from .indicator_manager import HigherTimeframeIndicatorManager


class StrategyIndicatorManager(IndicatorManager):
    """
    Enhanced indicator manager for strategies that supports higher timeframe indicators.
    
    This manager can handle:
    1. Regular indicators (calculated on the same timeframe as data)
    2. Higher timeframe indicators (calculated on higher timeframe and aligned to lower timeframe)
    """
    
    def __init__(self, use_parallel=False, max_workers=None):
        super().__init__(use_parallel, max_workers)
        self.htf_manager = HigherTimeframeIndicatorManager()
        self.has_htf_indicators = False
        
    def add_higher_timeframe_indicator(self, source_indicator, 
                                     higher_timeframe_interval='1h',
                                     lower_timeframe_interval='5m',
                                     alignment_method='forward_fill',
                                     shift_by_one_period=True):
        """
        Add a higher timeframe indicator to the strategy.
        
        Args:
            source_indicator: Base indicator to calculate on higher timeframe
            higher_timeframe_interval: Higher timeframe interval (e.g., '1h', '4h')
            lower_timeframe_interval: Lower timeframe interval (e.g., '5m', '1m')
            alignment_method: Method for aligning values
            shift_by_one_period: Whether to shift to avoid look-ahead bias
        """
        # Add to HTF manager
        self.htf_manager.add_higher_timeframe_indicator(
            source_indicator=source_indicator,
            higher_timeframe_interval=higher_timeframe_interval,
            lower_timeframe_interval=lower_timeframe_interval,
            alignment_method=alignment_method,
            shift_by_one_period=shift_by_one_period
        )
        
        # Create HTF indicator for tracking columns
        htf_indicator = HigherTimeframeIndicator(
            source_indicator=source_indicator,
            higher_timeframe_interval=higher_timeframe_interval,
            lower_timeframe_interval=lower_timeframe_interval,
            alignment_method=alignment_method,
            shift_by_one_period=shift_by_one_period
        )
        
        # Add to regular indicators list for column tracking
        self.indicators.append((htf_indicator, {}))
        column_names = htf_indicator.get_column_names()
        self.indicator_columns.extend(column_names)
        
        self.has_htf_indicators = True
        
    def calculate_all(self, data, higher_timeframe_data=None, append=True):
        """
        Calculate all indicators including higher timeframe indicators.
        
        Args:
            data: Lower timeframe data (DataFrame or dict)
            higher_timeframe_data: Higher timeframe data (DataFrame or dict)
            append: Whether to append to original data
            
        Returns:
            Enriched data with all indicators
        """
        # Separate regular and HTF indicators
        regular_indicators = []
        htf_indicators = []
        
        for indicator, kwargs in self.indicators:
            if isinstance(indicator, HigherTimeframeIndicator):
                htf_indicators.append((indicator, kwargs))
            else:
                regular_indicators.append((indicator, kwargs))
        
        # Calculate regular indicators first
        result = data.copy() if append else data
        
        for indicator, kwargs in regular_indicators:
            result = indicator.calculate(result, append=True, **kwargs)
        
        # Then calculate higher timeframe indicators if any
        if htf_indicators and higher_timeframe_data is not None:
            result = self.htf_manager.calculate_all(
                lower_timeframe_data=result,
                higher_timeframe_data=higher_timeframe_data,
                append=True
            )
        
        return result
    
    def get_htf_columns(self):
        """Get column names for higher timeframe indicators."""
        if self.has_htf_indicators:
            return self.htf_manager.get_indicator_columns()
        return []
    
    def has_higher_timeframe_indicators(self):
        """Check if manager has higher timeframe indicators."""
        return self.has_htf_indicators 