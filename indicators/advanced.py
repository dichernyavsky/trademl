"""
Advanced technical indicators module.

This module contains advanced technical indicators that combine multiple
basic indicators or provide specialized calculations.
"""

import pandas as pd
import numpy as np
from .base_indicator import BaseIndicator
from .support_resistance import SimpleSupportResistance
from .volatility import BollingerBands, ATRIndicator


class BreakoutGapATR(BaseIndicator):
    """Breakout‑Gap normalised ATR indicator.

    Computes two series:

    * **SupportGap**  = (Close − Support)    / ATR
    * **ResistanceGap** = (Close − Resistance) / ATR

    and optionally applies post‑scaling:

    Parameters
    ----------
    sr_lookback : int, default 20
        Lookback window for *SimpleSupportResistance* pivot search.
    atr_window : int, default 14
        ATR look‑back period.
    column : str, default "Close"
        Price column used for distance calculation.
    scale : {"raw", "clip", "tanh"}, default "raw"
        Choose how to post‑process the ratio.
        * "raw"   – leave as is.
        * "clip"  – clip to ±``clip_value``.
        * "tanh"  – clip, затем ``np.tanh(gap / clip_value)`` → (‑0.986 … 0.986).
    clip_value : float, default 5.0
        Threshold for clipping when *scale* ∈ {"clip", "tanh"}.
    shift_sr : bool, default False
        Shift SR lines by *sr_lookback* bars forward to avoid look‑ahead.
    """

    def __init__(self,
                 sr_lookback: int = 20,
                 atr_window: int = 14,
                 column: str = "Close",
                 scale: str = "raw",
                 clip_value: float = 5.0,
                 shift_sr: bool = False):
        self.sr_lookback = sr_lookback
        self.atr_window = atr_window
        self.column = column
        self.scale = scale.lower()
        self.clip_value = clip_value
        self.shift_sr = shift_sr
        super().__init__()

    # ------------------------------------------------------------------
    # BaseIndicator API
    # ------------------------------------------------------------------
    def get_column_names(self, **kwargs):
        base = f"BreakoutGapATR_{self.column}_{self.sr_lookback}_{self.atr_window}"
        return [f"{base}_Support", f"{base}_Resistance"]

    def _calculate_for_single_df(self,
                                 data: pd.DataFrame,
                                 append: bool = True,
                                 **kwargs):
        if data.empty:
            return data

        # 1) Support / Resistance
        sr_indicator = SimpleSupportResistance(lookback=self.sr_lookback)
        sr_data = sr_indicator.calculate(data, append=False)
        if self.shift_sr:
            sr_data = sr_data.shift(self.sr_lookback)

        sup_col = f"SimpleSR_{self.sr_lookback}_Support"
        res_col = f"SimpleSR_{self.sr_lookback}_Resistance"

        # 2) ATR
        atr_series = ATRIndicator(window=self.atr_window).calculate(
            data, append=False
        )[f"ATR_{self.atr_window}"]

        # 3) Raw gaps
        gap_support = (data[self.column] - sr_data[sup_col]) / (atr_series + 1e-9)
        gap_resist  = (data[self.column] - sr_data[res_col]) / (atr_series + 1e-9)

        # 4) Post‑scaling
        def _postprocess(arr: pd.Series):
            if self.scale == "raw":
                return arr
            arr = np.clip(arr, -self.clip_value, self.clip_value)
            return np.tanh(arr / (self.clip_value + 2)) if self.scale == "tanh" else arr

        gap_support = _postprocess(gap_support)
        gap_resist  = _postprocess(gap_resist)

        ind_data = {
            self.get_column_names()[0]: gap_support,
            self.get_column_names()[1]: gap_resist,
        }
        self.values = ind_data
        self.is_calculated = True

        return (self._append_to_df(data, ind_data)
                if append else self._create_indicator_df(data, ind_data))

class VWAPDistance(BaseIndicator):
    """
    VWAP Distance indicator.
    
    Calculates (Close - VWAP) / VWAP to measure price distance from VWAP
    as a percentage.
    """
    
    def __init__(self, column='Close'):
        """
        Initialize the VWAP Distance indicator.
        
        Args:
            column (str): Column to use for calculations
        """
        self.column = column
        super().__init__()
    
    def get_column_names(self, **kwargs):
        """Get column names for this indicator."""
        return [f'VWAPDistance_{self.column}']
    
    def _calculate_for_single_df(self, data, append=True, **kwargs):
        """
        Calculate VWAP Distance for a single DataFrame.
        
        Args:
            data: DataFrame with OHLCV data
            append: Whether to append to original DataFrame
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with VWAP Distance values
        """
        if data.empty:
            return data
        
        # Check if volume column exists
        if 'Volume' not in data.columns:
            raise ValueError("Volume column is required for VWAP calculation")
        
        # Calculate VWAP
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        volume_price = typical_price * data['Volume']
        cumulative_volume_price = volume_price.cumsum()
        cumulative_volume = data['Volume'].cumsum()
        vwap = cumulative_volume_price / (cumulative_volume + 1e-9)
        
        # Calculate VWAP distance
        vwap_distance = (data[self.column] - vwap) / (vwap + 1e-9)
        
        # Create indicator data
        indicator_data = {
            f'VWAPDistance_{self.column}': vwap_distance
        }
        
        self.values = indicator_data
        self.is_calculated = True
        
        return self._append_to_df(data, indicator_data) if append else self._create_indicator_df(data, indicator_data)


class BollingerBandwidth(BaseIndicator):
    """
    Bollinger Bandwidth indicator.
    
    Calculates (Upper - Lower) / Middle to measure the relative width
    of Bollinger Bands.
    """
    
    def __init__(self, window=20, num_std=2.0, column='Close'):
        """
        Initialize the Bollinger Bandwidth indicator.
        
        Args:
            window (int): Window size for Bollinger Bands calculation
            num_std (float): Number of standard deviations for the bands
            column (str): Column to use for calculations
        """
        self.window = window
        self.num_std = num_std
        self.column = column
        super().__init__()
    
    def get_column_names(self, **kwargs):
        """Get column names for this indicator."""
        return [f'BollingerBandwidth_{self.column}_{self.window}_std_{self.num_std}']
    
    def _calculate_for_single_df(self, data, append=True, **kwargs):
        """
        Calculate Bollinger Bandwidth for a single DataFrame.
        
        Args:
            data: DataFrame with OHLCV data
            append: Whether to append to original DataFrame
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with Bollinger Bandwidth values
        """
        if data.empty:
            return data
        
        # Calculate Bollinger Bands using the correct class
        from .volatility import BollingerBands
        bb_indicator = BollingerBands(window=self.window, num_std=self.num_std)
        bb_data = bb_indicator.calculate(data, append=False)
        
        # Get band values using the indicator's column names
        column_names = bb_indicator.get_column_names()
        middle_col = column_names[0]  # Middle band
        upper_col = column_names[1]   # Upper band  
        lower_col = column_names[2]   # Lower band
        
        # Calculate bandwidth
        bandwidth = (bb_data[upper_col] - bb_data[lower_col]) / (bb_data[middle_col] + 1e-9)
        
        # Create indicator data
        indicator_data = {
            f'BollingerBandwidth_{self.column}_{self.window}_std_{self.num_std}': bandwidth
        }
        
        self.values = indicator_data
        self.is_calculated = True
        
        return self._append_to_df(data, indicator_data) if append else self._create_indicator_df(data, indicator_data)


class NormalizedMomentum(BaseIndicator):
    """
    Normalized Momentum indicators.
    
    Calculates RSI/100, Stoch/100, and CCI/400 to normalize these indicators
    to specific ranges.
    """
    
    def __init__(self, rsi_window=14, stoch_window=14, cci_window=20, column='Close'):
        """
        Initialize the Normalized Momentum indicator.
        
        Args:
            rsi_window (int): Window size for RSI calculation
            stoch_window (int): Window size for Stochastic calculation
            cci_window (int): Window size for CCI calculation
            column (str): Column to use for calculations
        """
        self.rsi_window = rsi_window
        self.stoch_window = stoch_window
        self.cci_window = cci_window
        self.column = column
        super().__init__()
    
    def get_column_names(self, **kwargs):
        """Get column names for this indicator."""
        return [
            f'NormalizedRSI_{self.column}_{self.rsi_window}',
            f'NormalizedStoch_{self.column}_{self.stoch_window}',
            f'NormalizedCCI_{self.column}_{self.cci_window}'
        ]
    
    def _calculate_for_single_df(self, data, append=True, **kwargs):
        """
        Calculate Normalized Momentum indicators for a single DataFrame.
        
        Args:
            data: DataFrame with OHLCV data
            append: Whether to append to original DataFrame
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with Normalized Momentum values
        """
        if data.empty:
            return data
        
        # Calculate RSI
        rsi = self._calculate_rsi(data[self.column], self.rsi_window)
        normalized_rsi = rsi / 100.0  # Scale to [0, 1]
        
        # Calculate Stochastic
        stoch = self._calculate_stochastic(data, self.stoch_window)
        normalized_stoch = stoch / 100.0  # Scale to [0, 1]
        
        # Calculate CCI
        cci = self._calculate_cci(data, self.cci_window)
        normalized_cci = cci / 400.0  # Scale to [-1, 1] range
        
        # Create indicator data
        indicator_data = {
            f'NormalizedRSI_{self.column}_{self.rsi_window}': normalized_rsi,
            f'NormalizedStoch_{self.column}_{self.stoch_window}': normalized_stoch,
            f'NormalizedCCI_{self.column}_{self.cci_window}': normalized_cci
        }
        
        self.values = indicator_data
        self.is_calculated = True
        
        return self._append_to_df(data, indicator_data) if append else self._create_indicator_df(data, indicator_data)
    
    def _calculate_rsi(self, prices, window):
        """Calculate RSI."""
        delta = prices.diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        
        avg_gain = pd.Series(gain, index=prices.index).ewm(alpha=1/window, adjust=False).mean()
        avg_loss = pd.Series(loss, index=prices.index).ewm(alpha=1/window, adjust=False).mean()
        
        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - 100 / (1 + rs)
        return rsi
    
    def _calculate_stochastic(self, data, window):
        """Calculate Stochastic Oscillator %K."""
        low_min = data['Low'].rolling(window=window).min()
        high_max = data['High'].rolling(window=window).max()
        stoch = 100 * (data['Close'] - low_min) / (high_max - low_min + 1e-9)
        return stoch
    
    def _calculate_cci(self, data, window):
        """Calculate Commodity Channel Index."""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        sma_tp = typical_price.rolling(window=window).mean()
        
        # Vectorized MAD calculation
        mad_values = np.zeros(len(data))
        for i in range(window - 1, len(data)):
            window_tp = typical_price.iloc[i - window + 1:i + 1].values
            mad_values[i] = np.mean(np.abs(window_tp - window_tp.mean()))
        
        mad = pd.Series(mad_values, index=data.index)
        cci = (typical_price - sma_tp) / (0.015 * mad + 1e-9)
        return cci


class PercentRank(BaseIndicator):
    """
    Percent Rank indicator.
    
    Calculates the percentile rank of the current close price within
    a rolling window of N periods.
    """
    
    def __init__(self, window=20, column='Close'):
        """
        Initialize the Percent Rank indicator.
        
        Args:
            window (int): Window size for percentile calculation
            column (str): Column to use for calculations
        """
        self.window = window
        self.column = column
        super().__init__()
    
    def get_column_names(self, **kwargs):
        """Get column names for this indicator."""
        return [f'PercentRank_{self.column}_{self.window}']
    
    def _calculate_for_single_df(self, data, append=True, **kwargs):
        """
        Calculate Percent Rank for a single DataFrame.
        
        Args:
            data: DataFrame with OHLCV data
            append: Whether to append to original DataFrame
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with Percent Rank values
        """
        if data.empty:
            return data
        
        # Ultra-fast vectorized percent rank calculation using NumPy
        prices = data[self.column].values
        n = len(prices)
        percent_rank_values = np.full(n, np.nan)
        
        # Use NumPy's searchsorted for faster calculation
        for i in range(self.window - 1, n):
            window_prices = prices[i - self.window + 1:i + 1]
            current_price = window_prices[-1]
            # Use searchsorted to find position, then convert to rank
            rank = np.searchsorted(np.sort(window_prices), current_price, side='left') / len(window_prices)
            percent_rank_values[i] = rank
        
        # Create indicator data
        indicator_data = {
            f'PercentRank_{self.column}_{self.window}': pd.Series(percent_rank_values, index=data.index)
        }
        
        self.values = indicator_data
        self.is_calculated = True
        
        return self._append_to_df(data, indicator_data) if append else self._create_indicator_df(data, indicator_data) 