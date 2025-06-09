"""
Statistical event generators for financial data.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List, Tuple
from scipy import stats
from statsmodels.tsa.stattools import adfuller, coint
from arch import arch_model
from .base import EventGenerator, EventType

from indicators.volatility import BollingerBands



class ZScoreGenerator(EventGenerator):
    """
    Generates events when price z-scores exceed a threshold (Bollinger Bands approach).
    
    This generator identifies points where the price deviates significantly
    from its moving average, measured in standard deviations.
    """
    
    def __init__(self, close: pd.Series, window: int = 20, z_thresh: float = 2.0):
        """
        Parameters:
            close: Series of close prices
            window: Window size for z-score calculation
            z_thresh: Z-score threshold to generate an event
        """
        super().__init__(event_type=EventType.DIRECTION_SPECIFIC)
        self.close = close
        self.window = window
        self.z_thresh = z_thresh
        
        # Add indicators to calculate
        self.add_indicator(BollingerBands(window=window, num_std=2.0))
        
    def generate(self) -> pd.DataFrame:
        """
        Generate events when z-score exceeds threshold.
        
        Returns:
            DataFrame with event timestamps and direction column
        """
        # Prepare data for indicators
        data = pd.DataFrame({'Close': self.close})
        
        # Calculate indicators
        indicator_results = self.calculate_indicators(data)
        
        # Extract BB indicator results
        bb_indicator = indicator_results.get(self.indicators[0].name)
        if bb_indicator is None:
            raise ValueError("Failed to calculate Bollinger Bands indicator")
        
        # Extract z-scores (position relative to bands)
        bb_columns = bb_indicator.columns
        middle_band_col = next((col for col in bb_columns if 'middle' in col.lower()), None)
        upper_band_col = next((col for col in bb_columns if 'upper' in col.lower()), None)
        
        if not middle_band_col or not upper_band_col:
            raise ValueError("Could not identify Bollinger Bands columns")
        
        # Calculate z-scores
        middle_band = bb_indicator[middle_band_col]
        upper_band = bb_indicator[upper_band_col]
        std_dev = (upper_band - middle_band) / 2  # 2 std devs by default
        
        z_scores = (self.close - middle_band) / std_dev
        
        # Find events where absolute z-score exceeds threshold
        events = pd.Series(0, index=self.close.index)
        
        # Downward deviations (high z-score) generate expectations for mean reversion (down)
        # Upward deviations (negative z-score) generate expectations for mean reversion (up)
        events[z_scores > self.z_thresh] = -1  # Expected move: price will decrease
        events[z_scores < -self.z_thresh] = 1  # Expected move: price will increase
        
        # Keep only non-zero events
        events = events[events != 0]
        
        # Convert to DataFrame with direction column
        events_df = pd.DataFrame({'direction': events})
        
        self.events = events_df
        return self.events








class VolatilityBreakoutGenerator(EventGenerator):
    """
    Detects events based on volatility breakouts using Bollinger Bands
    and volume confirmation.
    """
    def __init__(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
        window: int = 20,
        num_std: float = 2.5,
        volume_factor: float = 1.5
    ):
        """
        Parameters:
            close: Series of close prices
            high: Series of high prices
            low: Series of low prices
            volume: Series of volume
            window: Lookback window for calculations
            num_std: Number of standard deviations for Bollinger Bands
            volume_factor: Volume increase factor required for confirmation
        """
        super().__init__(event_type=EventType.DIRECTION_SPECIFIC)
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume
        self.window = window
        self.num_std = num_std
        self.volume_factor = volume_factor
        
        # Add indicators for Bollinger Bands and Volume MA
        self.add_indicator(BollingerBands(window=window, num_std=num_std))
        from indicators.volume import VolumeMA
        self.add_indicator(VolumeMA(window=window, name="Volume_MA"))
        
    def generate(self) -> pd.DataFrame:
        """
        Generate volatility breakout events.
        
        Returns:
            DataFrame with event timestamps and direction column
        """
        # Prepare data for indicators
        data = pd.DataFrame({
            'Close': self.close,
            'High': self.high,
            'Low': self.low,
            'Volume': self.volume
        })
        
        # Calculate indicators
        indicator_results = self.calculate_indicators(data)
        
        # Extract BB indicator results
        bb_result = indicator_results.get(self.indicators[0].name)
        volume_ma_result = indicator_results.get(self.indicators[1].name)
        
        if bb_result is None or volume_ma_result is None:
            raise ValueError("Failed to calculate required indicators")
        
        # Extract bands from BB indicator
        bb_columns = bb_result.columns
        upper_band_col = next((col for col in bb_columns if 'upper' in col.lower()), None)
        lower_band_col = next((col for col in bb_columns if 'lower' in col.lower()), None)
        
        if not upper_band_col or not lower_band_col:
            raise ValueError("Could not identify Bollinger Bands columns")
        
        upper_band = bb_result[upper_band_col]
        lower_band = bb_result[lower_band_col]
        
        # Extract volume MA
        volume_ma_col = volume_ma_result.columns[0]  # First column is the MA
        volume_ma = volume_ma_result[volume_ma_col]
        volume_threshold = volume_ma * self.volume_factor
        
        # Find breakouts with volume confirmation
        breaks_up = (self.high > upper_band) & (self.volume > volume_threshold)
        breaks_down = (self.low < lower_band) & (self.volume > volume_threshold)
        
        # Return Series with direction
        events = pd.Series(0, index=self.close.index)
        events[breaks_up] = 1    # Upward breakouts
        events[breaks_down] = -1  # Downward breakouts
        
        # Keep only non-zero events
        events = events[events != 0]
        
        # Convert to DataFrame with direction column
        events_df = pd.DataFrame({'direction': events})
        
        self.events = events_df
        return self.events
