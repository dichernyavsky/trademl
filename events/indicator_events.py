"""
Event generators based on technical indicators.

These event generators use custom indicator classes to generate directional trade events.
"""

import pandas as pd
import numpy as np
from .base import EventGenerator, EventType
from ..indicators.volatility import *
from ..indicators.volume import *
from ..indicators.support_resistance import *

class BollingerBandsEventGenerator(EventGenerator):
    """
    Generates events based on Bollinger Bands indicator.
    
    Events are generated when price crosses or touches the bands, with
    expected direction based on mean reversion or breakout.
    """
    
    def __init__(self, window: int = 20, num_std: float = 2.0, 
                 mode: str = "reversal"):
        """
        Initialize the Bollinger Bands event generator.
        
        Parameters:
            window: Window size for moving average calculation
            num_std: Number of standard deviations for the bands
            mode: 'reversal' for mean reversion, 'breakout' for momentum strategy
        """
        super().__init__(event_type=EventType.DIRECTION_SPECIFIC)
        self.window = window
        self.num_std = num_std
        self.mode = mode
        
        # Add Bollinger Bands indicator
        self.add_indicator(BollingerBands(window=window, num_std=num_std))
    
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        indicator_results = self.calculate_indicators(data)
        middle_band = indicator_results['Bollinger_Bands_Middle']
        upper_band = indicator_results['Bollinger_Bands_Upper']
        lower_band = indicator_results['Bollinger_Bands_Lower']
        
        # Get close prices for signal generation
        self.close = data['Close']
        
        # Initialize events
        events = pd.Series(0, index=data.index)
        
        # Mean reversion strategy: buy when price hits lower band, sell when it hits upper band
        upper_touch = self.close >= upper_band
        lower_touch = self.close <= lower_band
        if self.mode == "reversal":
            # Set event directions
            events[upper_touch] = -1  # Sell signal - expect move down toward mean
            events[lower_touch] = 1   # Buy signal - expect move up toward mean
            
        elif self.mode == "breakout":
            # Breakout strategy: buy when price breaks above upper band, sell when it breaks below lower band
            events[upper_touch] = 1   # Buy signal - expect continued upward momentum
            events[lower_touch] = -1  # Sell signal - expect continued downward momentum
        
        # Keep only non-zero events
        events = events[events != 0]
        events_df = pd.DataFrame({'direction': events})
        
        self.events = events_df
        return self.events


class SimpleSREventGenerator(EventGenerator):
    """
    Generates events based on Support and Resistance levels.
    
    Events are generated when price approaches, breaks, or bounces from support and
    resistance levels with expected direction based on the event type.
    """
    
    def __init__(self, lookback: int = 20, mode: str = "breakout"):
        """
        Initialize the Support/Resistance event generator.
        
        Parameters:
            lookback: Number of bars to look before and after for pivot points
            mode: 'reversal' for bounces, 'breakout' for level breaks
        """
        super().__init__(event_type=EventType.DIRECTION_SPECIFIC)
        self.lookback = lookback
        self.mode = mode
        
        # Add Support/Resistance indicator
        self.add_indicator(SimpleSupportResistance(lookback=lookback))
    
    def generate(self, data: pd.DataFrame, keep_indicators: list = [], include_entry_price: bool = False) -> pd.DataFrame:
        """
        Generate breakout events based on support and resistance levels.

        Parameters:
            data: OHLCV DataFrame
            include_entry_price: Whether to include breakout price as the entry price in the output
            keep_indicators: List of indicator names to keep in the output
        Returns:
            DataFrame with event direction and optionally breakout price
        """
        # Calculate pivots and SR levels using the indicator directly
        sr_indicator = SimpleSupportResistance(lookback=self.lookback)
        enriched_data = sr_indicator.calculate(data, append=True)

        # Define column names for pivots and SR
        res_col = f"SimpleSR_{self.lookback}_Resistance"
        sup_col = f"SimpleSR_{self.lookback}_Support"

        # Check if columns exist
        if res_col not in enriched_data.columns or sup_col not in enriched_data.columns:
            raise ValueError(f"Required columns {res_col} and {sup_col} not found in data")

        # Prepare series
        high = enriched_data['High']
        low = enriched_data['Low']

        # Only breakout mode supported
        if self.mode != "breakout":
            raise ValueError(f"Mode '{self.mode}' is not supported yet.")

        # Forward-fill current SR levels
        enriched_data['res_current'] = enriched_data[res_col].ffill()
        enriched_data['sup_current'] = enriched_data[sup_col].ffill()

        # Detect end of SR levels
        res_end = enriched_data[res_col].shift(1).notna() & enriched_data[res_col].isna()
        sup_end = enriched_data[sup_col].shift(1).notna() & enriched_data[sup_col].isna()

        # Initialize direction series
        direction = pd.Series(0, index=enriched_data.index)
        direction.loc[res_end] = 1  # buy on resistance break
        direction.loc[sup_end] = -1  # sell on support break

        # Filter events
        event_mask = direction != 0
        result = pd.DataFrame({'direction': direction[event_mask]})

        if include_entry_price:
            levels = pd.Series(index=enriched_data.index, dtype=float)

            levels.loc[res_end] = enriched_data.loc[res_end, 'res_current']
            levels.loc[sup_end] = enriched_data.loc[sup_end, 'sup_current']
            result['entry_price'] = levels[event_mask]

        self.events = result
        return result