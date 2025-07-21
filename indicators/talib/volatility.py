"""
Volatility Indicators from TA-Lib

This module contains volatility indicators that can be normalized
for machine learning applications.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional, Tuple, Union

from .base_talib_indicator import BaseTALibIndicator


class ATR(BaseTALibIndicator):
    """
    Average True Range (ATR)
    
    ATR measures market volatility by decomposing the entire range of an asset
    price for that period. It can be normalized relative to price.
    
    Parameters:
        period (int): Number of periods for calculation (default: 14)
        normalize (bool): Whether to normalize values to 0-1 range (default: True)
    """
    
    def __init__(self, period: int = 14, normalize: bool = True):
        super().__init__()
        self.period = period
        self.normalize = normalize
    
    def get_column_names(self) -> List[str]:
        return [f'ATR_{self.period}']
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ATR
        
        Args:
            data: DataFrame with 'high', 'low', 'close' columns
            
        Returns:
            DataFrame with ATR column added
        """
        required_cols = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        result = data.copy()
        atr_values = talib.ATR(data['high'].values, data['low'].values,
                              data['close'].values, timeperiod=self.period)
        
        if self.normalize:
            # Normalize ATR relative to price (as percentage)
            atr_values = atr_values / data['close'].values
            # Clip to reasonable range and scale to 0-1
            atr_values = np.clip(atr_values, 0, 0.1) / 0.1
        
        result[f'ATR_{self.period}'] = atr_values
        
        return result


class BollingerBands(BaseTALibIndicator):
    """
    Bollinger Bands
    
    Bollinger Bands consist of a middle band (SMA) and two outer bands
    that are standard deviations away from the middle band.
    
    Parameters:
        period (int): Number of periods for SMA calculation (default: 20)
        nbdevup (float): Number of standard deviations for upper band (default: 2)
        nbdevdn (float): Number of standard deviations for lower band (default: 2)
        normalize (bool): Whether to normalize values to 0-1 range (default: True)
    """
    
    def __init__(self, period: int = 20, nbdevup: float = 2, nbdevdn: float = 2, normalize: bool = True):
        super().__init__()
        self.period = period
        self.nbdevup = nbdevup
        self.nbdevdn = nbdevdn
        self.normalize = normalize
    
    def get_column_names(self) -> List[str]:
        return [f'BB_UPPER_{self.period}_{self.nbdevup}_{self.nbdevdn}',
                f'BB_MIDDLE_{self.period}_{self.nbdevup}_{self.nbdevdn}',
                f'BB_LOWER_{self.period}_{self.nbdevup}_{self.nbdevdn}',
                f'BB_PERCENT_{self.period}_{self.nbdevup}_{self.nbdevdn}']
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands
        
        Args:
            data: DataFrame with 'close' column
            
        Returns:
            DataFrame with Bollinger Bands columns added
        """
        if 'close' not in data.columns:
            raise ValueError("DataFrame must contain 'close' column")
        
        result = data.copy()
        upper, middle, lower = talib.BBANDS(data['close'].values, timeperiod=self.period,
                                           nbdevup=self.nbdevup, nbdevdn=self.nbdevdn)
        
        # Calculate %B (position within bands)
        bb_percent = (data['close'].values - lower) / (upper - lower)
        
        if self.normalize:
            # %B is already 0-1, normalize other values relative to price
            price_mean = data['close'].mean()
            upper = (upper - price_mean) / price_mean
            middle = (middle - price_mean) / price_mean
            lower = (lower - price_mean) / price_mean
            
            # Scale to 0-1 range using sigmoid-like function
            upper = 1 / (1 + np.exp(-upper * 10))
            middle = 1 / (1 + np.exp(-middle * 10))
            lower = 1 / (1 + np.exp(-lower * 10))
        
        result[f'BB_UPPER_{self.period}_{self.nbdevup}_{self.nbdevdn}'] = upper
        result[f'BB_MIDDLE_{self.period}_{self.nbdevup}_{self.nbdevdn}'] = middle
        result[f'BB_LOWER_{self.period}_{self.nbdevup}_{self.nbdevdn}'] = lower
        result[f'BB_PERCENT_{self.period}_{self.nbdevup}_{self.nbdevdn}'] = bb_percent
        
        return result


class StandardDeviation(BaseTALibIndicator):
    """
    Standard Deviation
    
    Standard deviation measures the dispersion of price data from its mean.
    It can be normalized for machine learning applications.
    
    Parameters:
        period (int): Number of periods for calculation (default: 20)
        normalize (bool): Whether to normalize values to 0-1 range (default: True)
    """
    
    def __init__(self, period: int = 20, normalize: bool = True):
        super().__init__()
        self.period = period
        self.normalize = normalize
    
    def get_column_names(self) -> List[str]:
        return [f'STDDEV_{self.period}']
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Standard Deviation
        
        Args:
            data: DataFrame with 'close' column
            
        Returns:
            DataFrame with Standard Deviation column added
        """
        if 'close' not in data.columns:
            raise ValueError("DataFrame must contain 'close' column")
        
        result = data.copy()
        stddev_values = talib.STDDEV(data['close'].values, timeperiod=self.period)
        
        if self.normalize:
            # Normalize relative to price (as percentage)
            stddev_values = stddev_values / data['close'].values
            # Clip to reasonable range and scale to 0-1
            stddev_values = np.clip(stddev_values, 0, 0.1) / 0.1
        
        result[f'STDDEV_{self.period}'] = stddev_values
        
        return result 