"""
Oscillator Indicators from TA-Lib

This module contains oscillator indicators that typically range between 0-100
or have bounded ranges suitable for machine learning applications.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional, Tuple, Union

from .base_talib_indicator import BaseTALibIndicator


class RSI(BaseTALibIndicator):
    """
    Relative Strength Index (RSI)
    
    RSI is a momentum oscillator that measures the speed and magnitude of
    directional price movements. It oscillates between 0 and 100.
    
    Parameters:
        period (int): Number of periods for RSI calculation (default: 14)
        normalize (bool): Whether to normalize values to 0-1 range (default: True)
    """
    
    def __init__(self, period: int = 14, normalize: bool = True):
        super().__init__()
        self.period = period
        self.normalize = normalize
    
    def get_column_names(self) -> List[str]:
        return [f'RSI_{self.period}']
    
    def _calculate_talib(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """
        Calculate RSI indicator using TA-Lib.
        
        Args:
            data: DataFrame with TA-Lib standard column names
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with RSI values
        """
        self._validate_required_columns(data, ['close'])

        rsi_values = talib.RSI(data['close'].values, timeperiod=self.period)
        
        if self.normalize:
            # RSI is already 0-100, normalize to 0-1
            rsi_values = rsi_values / 100.0
        
        return {f'RSI_{self.period}': rsi_values}


class Stochastic(BaseTALibIndicator):
    """
    Stochastic Oscillator
    
    The Stochastic Oscillator is a momentum indicator that compares a closing
    price to its price range over a specific period. Both %K and %D oscillate
    between 0 and 100.
    
    Parameters:
        fastk_period (int): Period for %K calculation (default: 14)
        slowk_period (int): Period for %K smoothing (default: 3)
        slowd_period (int): Period for %D calculation (default: 3)
        normalize (bool): Whether to normalize values to 0-1 range (default: True)
    """
    
    def __init__(self, fastk_period: int = 14, slowk_period: int = 3, slowd_period: int = 3, normalize: bool = True):
        super().__init__()
        self.fastk_period = fastk_period
        self.slowk_period = slowk_period
        self.slowd_period = slowd_period
        self.normalize = normalize
    
    def get_column_names(self) -> List[str]:
        return [f'STOCH_K_{self.fastk_period}_{self.slowk_period}_{self.slowd_period}',
                f'STOCH_D_{self.fastk_period}_{self.slowk_period}_{self.slowd_period}']
    
    def _calculate_talib(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """
        Calculate Stochastic using TA-Lib.
        
        Args:
            data: DataFrame with TA-Lib standard column names
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with Stochastic values
        """
        self._validate_required_columns(data, ['high', 'low', 'close'])
        
        slowk, slowd = talib.STOCH(data['high'].values, data['low'].values, 
                                  data['close'].values, fastk_period=self.fastk_period,
                                  slowk_period=self.slowk_period, slowd_period=self.slowd_period)
        
        if self.normalize:
            # Stochastic is already 0-100, normalize to 0-1
            slowk = slowk / 100.0
            slowd = slowd / 100.0
        
        return {
            f'STOCH_K_{self.fastk_period}_{self.slowk_period}_{self.slowd_period}': slowk,
            f'STOCH_D_{self.fastk_period}_{self.slowk_period}_{self.slowd_period}': slowd
        }


class WilliamsR(BaseTALibIndicator):
    """
    Williams %R
    
    Williams %R is a momentum indicator that measures overbought/oversold levels.
    It oscillates between 0 and -100, where 0 is overbought and -100 is oversold.
    
    Parameters:
        period (int): Number of periods for calculation (default: 14)
        normalize (bool): Whether to normalize values to 0-1 range (default: True)
    """
    
    def __init__(self, period: int = 14, normalize: bool = True):
        super().__init__()
        self.period = period
        self.normalize = normalize
    
    def get_column_names(self) -> List[str]:
        return [f'WILLR_{self.period}']
    
    def _calculate_talib(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """
        Calculate Williams %R using TA-Lib.
        
        Args:
            data: DataFrame with TA-Lib standard column names
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with Williams %R values
        """
        self._validate_required_columns(data, ['high', 'low', 'close'])
        
        willr_values = talib.WILLR(data['high'].values, data['low'].values, 
                                 data['close'].values, timeperiod=self.period)
        
        if self.normalize:
            # Williams %R is 0 to -100, normalize to 0-1 (invert and scale)
            willr_values = (willr_values + 100) / 100.0
        
        return {f'WILLR_{self.period}': willr_values}


class CCI(BaseTALibIndicator):
    """
    Commodity Channel Index (CCI)
    
    CCI measures the current price level relative to an average price level
    over a given period. It oscillates around zero, typically between ±100-200.
    
    Parameters:
        period (int): Number of periods for calculation (default: 14)
        normalize (bool): Whether to normalize values to 0-1 range (default: True)
    """
    
    def __init__(self, period: int = 14, normalize: bool = True):
        super().__init__()
        self.period = period
        self.normalize = normalize
    
    def get_column_names(self) -> List[str]:
        return [f'CCI_{self.period}']
    
    def _calculate_for_single_df(self, data: pd.DataFrame, append: bool = True, **kwargs):
        """
        Calculate CCI for a single DataFrame.
        
        Args:
            data: DataFrame with 'high', 'low', 'close' columns
            append: Whether to append to original DataFrame
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with CCI column added
        """
        required_cols = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        cci_values = talib.CCI(data['high'].values, data['low'].values, 
                              data['close'].values, timeperiod=self.period)
        
        if self.normalize:
            # CCI typically ranges ±100-200, normalize to 0-1 using sigmoid-like scaling
            cci_values = 1 / (1 + np.exp(-cci_values / 100))
        
        # Create indicator data
        indicator_data = {f'CCI_{self.period}': cci_values}
        
        # Store values for consistency with BaseIndicator
        self.values = indicator_data
        self.is_calculated = True
        
        return self._append_to_df(data, indicator_data) if append else self._create_indicator_df(data, indicator_data)


class ROC(BaseTALibIndicator):
    """
    Rate of Change (ROC)
    
    ROC measures the percentage change in price over a specified period.
    It can be normalized for machine learning applications.
    
    Parameters:
        period (int): Number of periods for calculation (default: 10)
        normalize (bool): Whether to normalize values to 0-1 range (default: True)
    """
    
    def __init__(self, period: int = 10, normalize: bool = True):
        super().__init__()
        self.period = period
        self.normalize = normalize
    
    def get_column_names(self) -> List[str]:
        return [f'ROC_{self.period}']
    
    def _calculate_for_single_df(self, data: pd.DataFrame, append: bool = True, **kwargs):
        """
        Calculate ROC for a single DataFrame.
        
        Args:
            data: DataFrame with 'close' column
            append: Whether to append to original DataFrame
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with ROC column added
        """
        if 'close' not in data.columns:
            raise ValueError("DataFrame must contain 'close' column")
        
        roc_values = talib.ROC(data['close'].values, timeperiod=self.period)
        
        if self.normalize:
            # ROC can be any value, normalize using sigmoid-like scaling
            roc_values = 1 / (1 + np.exp(-roc_values / 10))
        
        # Create indicator data
        indicator_data = {f'ROC_{self.period}': roc_values}
        
        # Store values for consistency with BaseIndicator
        self.values = indicator_data
        self.is_calculated = True
        
        return self._append_to_df(data, indicator_data) if append else self._create_indicator_df(data, indicator_data)


class Momentum(BaseTALibIndicator):
    """
    Momentum
    
    Momentum measures the rate of change in price over a specified period.
    It can be normalized for machine learning applications.
    
    Parameters:
        period (int): Number of periods for calculation (default: 10)
        normalize (bool): Whether to normalize values to 0-1 range (default: True)
    """
    
    def __init__(self, period: int = 10, normalize: bool = True):
        super().__init__()
        self.period = period
        self.normalize = normalize
    
    def get_column_names(self) -> List[str]:
        return [f'MOM_{self.period}']
    
    def _calculate_talib(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """
        Calculate Momentum using TA-Lib.
        
        Args:
            data: DataFrame with TA-Lib standard column names
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with Momentum values
        """
        self._validate_required_columns(data, ['close'])
        
        mom_values = talib.MOM(data['close'].values, timeperiod=self.period)
        
        if self.normalize:
            # Momentum can be any value, normalize using sigmoid-like scaling
            mom_values = 1 / (1 + np.exp(-mom_values / (data['close'].mean() * 0.01)))
        
        return {f'MOM_{self.period}': mom_values}


class MACD(BaseTALibIndicator):
    """
    Moving Average Convergence Divergence (MACD)
    
    MACD is a trend-following momentum indicator that shows the relationship
    between two moving averages of a security's price.
    
    Parameters:
        fastperiod (int): Fast EMA period (default: 12)
        slowperiod (int): Slow EMA period (default: 26)
        signalperiod (int): Signal line period (default: 9)
        normalize (bool): Whether to normalize values to 0-1 range (default: True)
    """
    
    def __init__(self, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9, normalize: bool = True):
        super().__init__()
        self.fastperiod = fastperiod
        self.slowperiod = slowperiod
        self.signalperiod = signalperiod
        self.normalize = normalize
    
    def get_column_names(self) -> List[str]:
        return [f'MACD_{self.fastperiod}_{self.slowperiod}_{self.signalperiod}',
                f'MACD_SIGNAL_{self.fastperiod}_{self.slowperiod}_{self.signalperiod}',
                f'MACD_HIST_{self.fastperiod}_{self.slowperiod}_{self.signalperiod}']
    
    def _calculate_talib(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """
        Calculate MACD using TA-Lib.
        
        Args:
            data: DataFrame with TA-Lib standard column names
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with MACD values
        """
        self._validate_required_columns(data, ['close'])
        
        macd, signal, hist = talib.MACD(data['close'].values, fastperiod=self.fastperiod,
                                       slowperiod=self.slowperiod, signalperiod=self.signalperiod)
        
        if self.normalize:
            # Normalize MACD values using sigmoid-like scaling
            price_range = data['close'].max() - data['close'].min()
            scale_factor = price_range * 0.01
            
            macd = 1 / (1 + np.exp(-macd / scale_factor))
            signal = 1 / (1 + np.exp(-signal / scale_factor))
            hist = 1 / (1 + np.exp(-hist / scale_factor))
        
        return {
            f'MACD_{self.fastperiod}_{self.slowperiod}_{self.signalperiod}': macd,
            f'MACD_SIGNAL_{self.fastperiod}_{self.slowperiod}_{self.signalperiod}': signal,
            f'MACD_HIST_{self.fastperiod}_{self.slowperiod}_{self.signalperiod}': hist
        }


class ADX(BaseTALibIndicator):
    """
    Average Directional Index (ADX)
    
    ADX measures the strength of a trend, regardless of direction.
    It ranges from 0 to 100, where values above 25 indicate a strong trend.
    
    Parameters:
        period (int): Number of periods for calculation (default: 14)
        normalize (bool): Whether to normalize values to 0-1 range (default: True)
    """
    
    def __init__(self, period: int = 14, normalize: bool = True):
        super().__init__()
        self.period = period
        self.normalize = normalize
    
    def get_column_names(self) -> List[str]:
        return [f'ADX_{self.period}', f'ADX_PLUS_DI_{self.period}', f'ADX_MINUS_DI_{self.period}']
    
    def _calculate_talib(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """
        Calculate ADX using TA-Lib.
        
        Args:
            data: DataFrame with TA-Lib standard column names
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with ADX values
        """
        self._validate_required_columns(data, ['high', 'low', 'close'])
        
        adx = talib.ADX(data['high'].values, data['low'].values,
                       data['close'].values, timeperiod=self.period)
        plus_di = talib.PLUS_DI(data['high'].values, data['low'].values,
                               data['close'].values, timeperiod=self.period)
        minus_di = talib.MINUS_DI(data['high'].values, data['low'].values,
                                 data['close'].values, timeperiod=self.period)
        
        if self.normalize:
            # ADX is already 0-100, normalize to 0-1
            adx = adx / 100.0
            plus_di = plus_di / 100.0
            minus_di = minus_di / 100.0
        
        return {
            f'ADX_{self.period}': adx,
            f'ADX_PLUS_DI_{self.period}': plus_di,
            f'ADX_MINUS_DI_{self.period}': minus_di
        }


class Aroon(BaseTALibIndicator):
    """
    Aroon Indicator
    
    Aroon measures the time between highs and lows over a time period.
    Both Aroon Up and Down range from 0 to 100.
    
    Parameters:
        period (int): Number of periods for calculation (default: 14)
        normalize (bool): Whether to normalize values to 0-1 range (default: True)
    """
    
    def __init__(self, period: int = 14, normalize: bool = True):
        super().__init__()
        self.period = period
        self.normalize = normalize
    
    def get_column_names(self) -> List[str]:
        return [f'AROON_UP_{self.period}', f'AROON_DOWN_{self.period}']
    
    def _calculate_talib(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """
        Calculate Aroon using TA-Lib.
        
        Args:
            data: DataFrame with TA-Lib standard column names
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with Aroon values
        """
        self._validate_required_columns(data, ['high', 'low'])
        
        aroon_up, aroon_down = talib.AROON(data['high'].values, data['low'].values, timeperiod=self.period)
        
        if self.normalize:
            # Aroon is already 0-100, normalize to 0-1
            aroon_up = aroon_up / 100.0
            aroon_down = aroon_down / 100.0
        
        return {
            f'AROON_UP_{self.period}': aroon_up,
            f'AROON_DOWN_{self.period}': aroon_down
        }


class TRIX(BaseTALibIndicator):
    """
    TRIX
    
    TRIX is a momentum oscillator that shows the percentage rate of change
    of a triple exponentially smoothed moving average.
    
    Parameters:
        period (int): Number of periods for calculation (default: 30)
        normalize (bool): Whether to normalize values to 0-1 range (default: True)
    """
    
    def __init__(self, period: int = 30, normalize: bool = True):
        super().__init__()
        self.period = period
        self.normalize = normalize
    
    def get_column_names(self) -> List[str]:
        return [f'TRIX_{self.period}']
    
    def _calculate_talib(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """
        Calculate TRIX using TA-Lib.
        
        Args:
            data: DataFrame with TA-Lib standard column names
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with TRIX values
        """
        self._validate_required_columns(data, ['close'])
        
        trix_values = talib.TRIX(data['close'].values, timeperiod=self.period)
        
        if self.normalize:
            # TRIX can be any value, normalize using sigmoid-like scaling
            trix_values = 1 / (1 + np.exp(-trix_values * 100))
        
        return {f'TRIX_{self.period}': trix_values} 