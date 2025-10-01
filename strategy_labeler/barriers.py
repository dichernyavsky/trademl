import pandas as pd
import numpy as np
from .base_barrier import Barrier

# ----- BASE BARRIER CLASS ------------------------------------------------------------------
class SimpleVolatilityBarrier(Barrier):
    """
    Simple volatility-based barrier strategy.
    Calculates PT/SL barriers based on price volatility.
    """
    
    def __init__(self, window=20, multiplier=[2, 2], min_ret=0.001, hold_periods=50):
        """
        Initialize SimpleVolatilityBarrier.
        
        Args:
            window (int): Lookback window for volatility calculation
            multiplier (list): [pt_multiplier, sl_multiplier] for volatility scaling
            min_ret (float): Minimum return threshold for barriers
            hold_periods (int): Number of bars/candles to hold for vertical barrier
        """
        super().__init__(hold_periods=hold_periods, window=window, multiplier=multiplier, min_ret=min_ret)
        self.window = window
        self.multiplier = multiplier
        self.min_ret = min_ret
    
    def _calculate_horizontal_barriers(self, events, data, **kwargs):
        """
        Calculate horizontal barriers using volatility-based approach.
        Accounts for position direction (long/short).
        
        Args:
            events: DataFrame with events (must have 'direction' column)
            data: DataFrame with OHLCV data
            **kwargs: Additional parameters (can override instance variables)
            
        Returns:
            DataFrame: Events with added 'pt' and 'sl' columns
        """
        # Use kwargs if provided, otherwise use instance variables
        window = kwargs.get('window', self.window)
        multiplier = kwargs.get('multiplier', self.multiplier)
        min_ret = kwargs.get('min_ret', self.min_ret)
        
        # Get close prices
        close = data['Close']
        
        # Calculate simple returns
        rets = close.pct_change()
        
        # Calculate volatility using standard deviation
        vol = rets.rolling(window=window).std()
        
        # Get volatility at event times
        event_vol = vol[events.index]
        
        # Calculate target sizes (maximum of volatility-based and minimum threshold)
        pt_target = np.maximum(multiplier[0] * event_vol, min_ret)
        sl_target = np.maximum(multiplier[1] * event_vol, min_ret)
        
        # Calculate barriers relative to entry price
        # Use entry_price from events if available, otherwise use close price at event times
        if 'entry_price' in events.columns:
            entry_prices = events['entry_price']
            print(f"Using entry_price from events for barrier calculation")
        else:
            entry_prices = close[events.index]
            print(f"Using close price at event times for barrier calculation")
        
        # Create a copy of events to avoid modifying original
        result = events.copy()
        #result['target'] = pt_target  # Keep original target column with PT target
        
        # Calculate barriers based on position direction
        for i, (event_time, event_row) in enumerate(result.iterrows()):
            direction = event_row['direction']
            entry_price = entry_prices.iloc[i] if hasattr(entry_prices, 'iloc') else entry_prices[event_time]
            pt_size = pt_target.iloc[i] if hasattr(pt_target, 'iloc') else pt_target[event_time]
            sl_size = sl_target.iloc[i] if hasattr(sl_target, 'iloc') else sl_target[event_time]
            
            if direction == 1:  # Long position
                # For longs: PT above entry, SL below entry
                result.at[event_time, 'pt'] = entry_price * (1 + pt_size)
                result.at[event_time, 'sl'] = entry_price * (1 - sl_size)
            elif direction == -1:  # Short position
                # For shorts: PT below entry, SL above entry
                result.at[event_time, 'pt'] = entry_price * (1 - pt_size)
                result.at[event_time, 'sl'] = entry_price * (1 + sl_size)
            else:
                # Neutral event - use default long position barriers
                result.at[event_time, 'pt'] = entry_price * (1 + pt_size)
                result.at[event_time, 'sl'] = entry_price * (1 - sl_size)
        
        return result