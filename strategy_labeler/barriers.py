import pandas as pd
import numpy as np

# ----- BASE BARRIER CLASS ------------------------------------------------------------------
class Barrier:
    """
    Base class for all barrier strategies.
    Each barrier strategy must implement calculate_barriers method.
    Also handles vertical barriers (t1) and trade generation by default.
    """
    
    def __init__(self, hold_periods=50, **kwargs):
        """
        Initialize barrier strategy with parameters.
        
        Args:
            hold_periods (int): Number of bars/candles to hold for vertical barrier
            **kwargs: Strategy-specific parameters
        """
        self.params = kwargs
        self.hold_periods = hold_periods
    
    def calculate_barriers(self, events, data, **kwargs):
        """
        Calculate profit-taking and stop-loss barriers for events.
        Also adds vertical barrier (t1) by default.
        
        Args:
            events: DataFrame with events (must have index as timestamps)
            data: DataFrame with OHLCV data or dict of DataFrames
            **kwargs: Additional parameters
            
        Returns:
            DataFrame: Events with added 'pt', 'sl', and 't1' columns
        """
        # First calculate horizontal barriers (PT/SL)
        events = self._calculate_horizontal_barriers(events, data, **kwargs)
        
        # Then add vertical barrier (t1)
        events = self._add_vertical_barrier(events, data, **kwargs)
        
        return events
    
    def generate_trades(self, events, data, trailing_stop=False, trailing_pct=None, 
                       save_trail=False, use_hl=True, **kwargs):
        """
        Generate trades from events with barriers.
        This method handles both fixed and trailing stops.
        
        Args:
            events: DataFrame with events and barriers
            data: DataFrame with OHLCV data
            trailing_stop: If True, use trailing stop instead of fixed stop
            trailing_pct: Trailing distance as fraction of stop loss distance (e.g., 0.5 = 50% of SL distance) (if None, uses 0.5 = 50%)
            save_trail: If True, save the dynamic stop path
            use_hl: If True, use high/low for barrier touches
            **kwargs: Additional parameters
            
        Returns:
            DataFrame: Trades with entry/exit information
        """
        if trailing_stop:
            return self._generate_trades_with_trailing_stop(
                events, data, trailing_pct, save_trail, use_hl, **kwargs
            )
        else:
            return self._generate_trades_with_fixed_stop(
                events, data, use_hl, **kwargs
            )
    
    def _calculate_horizontal_barriers(self, events, data, **kwargs):
        """
        Calculate horizontal barriers (PT/SL).
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _calculate_horizontal_barriers")
    
    def _add_vertical_barrier(self, events, data, **kwargs):
        """
        Add vertical barrier (t1) to events.
        
        Args:
            events: DataFrame with events
            data: DataFrame with OHLCV data
            **kwargs: Additional parameters
            
        Returns:
            DataFrame: Events with added 't1' column
        """
        hold_periods = kwargs.get('hold_periods', self.hold_periods)
        
        # Create a copy to avoid modifying original
        result = events.copy()
        result['t1'] = None
        
        close = data['Close']
        
        for i, event_time in enumerate(result.index):
            try:
                # Find position in data index
                event_pos = close.index.get_loc(event_time)
                # Add hold_periods positions
                target_pos = min(event_pos + hold_periods, len(close.index) - 1)
                result.iloc[i, result.columns.get_loc('t1')] = close.index[target_pos]
            except (KeyError, IndexError):
                # If event time not found or out of bounds, use last available time
                result.iloc[i, result.columns.get_loc('t1')] = close.index[-1]
        
        return result
    
    def _generate_trades_with_fixed_stop(self, events, data, use_hl=True, **kwargs):
        """
        Generate trades using fixed stop-loss and take-profit barriers.
        
        Args:
            events: DataFrame with events and barriers
            data: DataFrame with OHLCV data
            use_hl: If True, use high/low for barrier touches
            **kwargs: Additional parameters
            
        Returns:
            DataFrame: Trades with entry/exit information
        """
        out = events.copy()
        out['bin'] = 0  # default label
        out['exit_time'] = pd.NaT
        out['exit_price'] = np.nan
        
        # Add entry price column if not already present
        if 'entry_price' not in out.columns:
            out['entry_price'] = np.nan
        
        # Check if we have OHLCV data
        is_ohlcv = isinstance(data, pd.DataFrame) and 'High' in data.columns and 'Low' in data.columns
        close_data = data['Close'] if is_ohlcv else data
        
        for loc, barrier_time in out['t1'].items():
            # Set entry price if not already set
            if pd.isna(out.at[loc, 'entry_price']) and loc in close_data.index:
                out.at[loc, 'entry_price'] = close_data.loc[loc]
            
            # Get price path from entry to barrier
            if is_ohlcv:
                price_path = data.loc[loc:barrier_time]
            else:
                price_path = data.loc[loc:barrier_time]
                
            if len(price_path) == 0:
                continue
            
            # Get direction and barriers
            direction = out.at[loc, 'direction']
            pt_barrier = out.at[loc, 'pt']
            sl_barrier = out.at[loc, 'sl']
            
            # Adjust barriers based on direction
            if direction == 1:  # Long position
                # For longs: PT above entry, SL below entry
                upper_barrier = pt_barrier
                lower_barrier = sl_barrier
            elif direction == -1:  # Short position
                # For shorts: PT below entry, SL above entry
                upper_barrier = sl_barrier
                lower_barrier = pt_barrier
            else:
                continue  # Skip neutral events
            
            # Check barrier touches
            if is_ohlcv and use_hl:
                touched_upper = price_path['High'][price_path['High'] >= upper_barrier]
                touched_lower = price_path['Low'][price_path['Low'] <= lower_barrier]
            else:
                close_path = price_path['Close'] if is_ohlcv else price_path
                touched_upper = close_path[close_path >= upper_barrier]
                touched_lower = close_path[close_path <= lower_barrier]
            
            # Determine which barrier was hit first
            if not touched_upper.empty and (touched_lower.empty or touched_upper.index[0] <= touched_lower.index[0]):
                out.at[loc, 'bin'] = 1 if direction == 1 else -1  # PT hit
                out.at[loc, 'exit_time'] = touched_upper.index[0]
                out.at[loc, 'exit_price'] = upper_barrier
            elif not touched_lower.empty and (touched_upper.empty or touched_lower.index[0] < touched_upper.index[0]):
                out.at[loc, 'bin'] = -1 if direction == 1 else 1  # SL hit
                out.at[loc, 'exit_time'] = touched_lower.index[0]
                out.at[loc, 'exit_price'] = lower_barrier
            else:
                out.at[loc, 'bin'] = 0  # Time exit
                out.at[loc, 'exit_time'] = barrier_time
                out.at[loc, 'exit_price'] = close_data.loc[barrier_time]
        
        return out
    
    def _generate_trades_with_trailing_stop(self, events, data, trailing_pct=None, 
                                          save_trail=False, use_hl=True, **kwargs):
        """
        Generate trades using trailing stop-loss.
        
        Args:
            events: DataFrame with events and barriers
            data: DataFrame with OHLCV data
            trailing_pct: Trailing distance as fraction of stop loss distance (e.g., 0.5 = 50% of SL distance) (if None, uses 0.5 = 50%)
            save_trail: If True, save the dynamic stop path
            use_hl: If True, use high/low for barrier touches
            **kwargs: Additional parameters
            
        Returns:
            DataFrame: Trades with entry/exit information
        """
        out = events.copy()
        out['bin'] = 0
        out['exit_time'] = pd.NaT
        out['exit_price'] = np.nan
        
        if save_trail:
            out['dynamic_stop_path'] = None
        
        # Add entry price column if not already present
        if 'entry_price' not in out.columns:
            out['entry_price'] = np.nan
        
        # Check if we have OHLCV data
        is_ohlcv = isinstance(data, pd.DataFrame) and 'High' in data.columns and 'Low' in data.columns
        close_data = data['Close'] if is_ohlcv else data
        
        for loc, barrier_time in out['t1'].items():
            # Set entry price if not already set
            if pd.isna(out.at[loc, 'entry_price']) and loc in close_data.index:
                out.at[loc, 'entry_price'] = close_data.loc[loc]
            
            # Get price path from entry to barrier
            if is_ohlcv:
                price_path = data.loc[loc:barrier_time]
            else:
                price_path = data.loc[loc:barrier_time]
                
            if len(price_path) == 0:
                continue
            
            # Get direction and barriers
            direction = out.at[loc, 'direction']
            pt_barrier = out.at[loc, 'pt']
            sl_barrier = out.at[loc, 'sl']
            entry_price = out.at[loc, 'entry_price']
            
            # Calculate trailing percentage based on SL distance if not provided
            if trailing_pct is None:
                trailing_pct = 0.5  # Default: 50% of SL distance
            
            # Calculate trailing stop based on direction
            if direction == 1:  # Long position
                # For longs: trail below the price
                prices_for_trail = price_path['Close'].values
                cum_max = np.maximum.accumulate(prices_for_trail)
                
                # Calculate trailing distance as fraction of stop loss distance
                sl_distance = entry_price - sl_barrier  # Distance from entry to SL
                trailing_distance = sl_distance * trailing_pct  # Fraction of that distance
                dynamic_sl = cum_max - trailing_distance  # Trail below cum_max by this distance
                
                # Check PT and trailing stop
                if is_ohlcv and use_hl:
                    pt_touched = price_path['High'][price_path['High'] >= pt_barrier]
                    sl_touched = price_path['Low'][price_path['Low'] <= dynamic_sl]
                else:
                    pt_touched = price_path['Close'][price_path['Close'] >= pt_barrier]
                    sl_touched = price_path['Close'][price_path['Close'] <= dynamic_sl]
                    
            elif direction == -1:  # Short position
                # For shorts: trail above the price
                prices_for_trail = price_path['Close'].values
                cum_min = np.minimum.accumulate(prices_for_trail)
                
                # Calculate trailing distance as fraction of stop loss distance
                sl_distance = sl_barrier - entry_price  # Distance from entry to SL
                trailing_distance = sl_distance * trailing_pct  # Fraction of that distance
                dynamic_sl = cum_min + trailing_distance  # Trail above cum_min by this distance
                
                # Check PT and trailing stop
                if is_ohlcv and use_hl:
                    pt_touched = price_path['Low'][price_path['Low'] <= pt_barrier]
                    sl_touched = price_path['High'][price_path['High'] >= dynamic_sl]
                else:
                    pt_touched = price_path['Close'][price_path['Close'] <= pt_barrier]
                    sl_touched = price_path['Close'][price_path['Close'] >= dynamic_sl]
            else:
                continue
            
            # Determine which barrier was hit first
            if not pt_touched.empty and (sl_touched.empty or pt_touched.index[0] <= sl_touched.index[0]):
                out.at[loc, 'bin'] = 1 if direction == 1 else -1  # PT hit
                out.at[loc, 'exit_time'] = pt_touched.index[0]
                out.at[loc, 'exit_price'] = pt_barrier
            elif not sl_touched.empty and (pt_touched.empty or sl_touched.index[0] < pt_touched.index[0]):
                out.at[loc, 'bin'] = -1 if direction == 1 else 1  # SL hit
                out.at[loc, 'exit_time'] = sl_touched.index[0]
                # Get the dynamic stop level at exit time
                exit_idx = list(price_path.index).index(out.at[loc, 'exit_time'])
                actual_sl_value = dynamic_sl[exit_idx]
                out.at[loc, 'exit_price'] = actual_sl_value
                # Update the sl barrier to show the actual stop loss value at exit
                out.at[loc, 'sl'] = actual_sl_value
            else:
                out.at[loc, 'bin'] = 0  # Time exit
                out.at[loc, 'exit_time'] = barrier_time
                out.at[loc, 'exit_price'] = close_data.loc[barrier_time]
            
            # Save trailing path if requested
            if save_trail:
                trail_series = pd.Series(dynamic_sl, index=price_path.index)
                out.at[loc, 'dynamic_stop_path'] = trail_series
        
        return out
    
    def validate_config(self):
        """
        Validate barrier configuration.
        Override in subclasses if needed.
        """
        return True


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
        result['target'] = pt_target  # Keep original target column with PT target
        
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


class SimpleFixedBarrier(Barrier):
    """
    Simple fixed percentage barrier strategy.
    Calculates PT/SL barriers based on fixed percentage multipliers.
    """
    
    def __init__(self, tp_multiplier=0.05, sl_multiplier=0.03, hold_periods=50):
        """
        Initialize SimpleFixedBarrier.
        
        Args:
            tp_multiplier (float): Fixed percentage for take profit (e.g., 0.05 = 5%)
            sl_multiplier (float): Fixed percentage for stop loss (e.g., 0.03 = 3%)
            hold_periods (int): Number of bars/candles to hold for vertical barrier
        """
        super().__init__(hold_periods=hold_periods, tp_multiplier=tp_multiplier, sl_multiplier=sl_multiplier)
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
    
    def _calculate_horizontal_barriers(self, events, data, **kwargs):
        """
        Calculate horizontal barriers using fixed percentage approach.
        Accounts for position direction (long/short).
        
        Args:
            events: DataFrame with events (must have 'direction' column)
            data: DataFrame with OHLCV data
            **kwargs: Additional parameters (can override instance variables)
            
        Returns:
            DataFrame: Events with added 'pt' and 'sl' columns
        """
        # Use kwargs if provided, otherwise use instance variables
        tp_multiplier = kwargs.get('tp_multiplier', self.tp_multiplier)
        sl_multiplier = kwargs.get('sl_multiplier', self.sl_multiplier)
        
        # Get close prices
        close = data['Close']
        
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
        
        # Calculate barriers based on position direction
        for i, (event_time, event_row) in enumerate(result.iterrows()):
            direction = event_row['direction']
            entry_price = entry_prices.iloc[i] if hasattr(entry_prices, 'iloc') else entry_prices[event_time]
            
            if direction == 1:  # Long position
                # For longs: PT above entry, SL below entry
                result.at[event_time, 'pt'] = entry_price * (1 + tp_multiplier)
                result.at[event_time, 'sl'] = entry_price * (1 - sl_multiplier)
            elif direction == -1:  # Short position
                # For shorts: PT below entry, SL above entry
                result.at[event_time, 'pt'] = entry_price * (1 - tp_multiplier)
                result.at[event_time, 'sl'] = entry_price * (1 + sl_multiplier)
            else:
                # Neutral event - use default long position barriers
                result.at[event_time, 'pt'] = entry_price * (1 + tp_multiplier)
                result.at[event_time, 'sl'] = entry_price * (1 - sl_multiplier)
        
        result['target'] = tp_multiplier  # Target is the TP percentage
        
        return result


# ----- VERTICAL BARRIERS ------------------------------------------------------------------
def SimpleVerticalBarrier(close, tEvents, numBars=1, barType='time'):
    """
    Compute vertical barriers that work with different bar sampling methods.
    
    Parameters:
        close: pandas Series of close prices
        tEvents: pandas Series/DatetimeIndex of event start times
        numBars: number of bars to look forward
        barType: string indicating the bar sampling method:
                'time' - traditional time-based bars
                'volume' - volume-based bars
                'dollar' - dollar-based bars
                'tick' - tick-based bars
    
    Returns:
        pd.Series with vertical barrier timestamps
    """
    if barType == 'time':
        # Original time-based implementation
        t1 = close.index.searchsorted(tEvents + pd.Timedelta(days=numBars))
    else:
        # For volume/dollar/tick bars, simply move forward N bars
        t1 = pd.Series(index=tEvents, dtype=object)
        for i, event in enumerate(tEvents):
            future_idx = close.index.get_loc(event) + numBars
            if future_idx < len(close.index):
                t1.iloc[i] = close.index[future_idx]
            
    # Filter out barriers beyond the last observation
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=tEvents[:t1.shape[0]])
    return t1



# ----- HORIZONTAL BARRIERS ------------------------------------------------------------------

def ptsl_simple(events, close, window=20, multiplier=[2, 2], min_ret=0.001):
    """
    Add profit-taking and stop-loss barriers with dynamic sizing based on volatility.
    
    Parameters:
        events: DataFrame with index representing event start times
        close: pandas Series of close prices
        window: lookback window for volatility calculation
        multiplier: list of [pt_multiplier, sl_multiplier] to multiply volatility by for target sizes
        min_ret: minimum return threshold for barriers
    
    Returns:
        DataFrame with added columns:
        - 'pt': profit-taking barrier price
        - 'sl': stop-loss barrier price
        - 'target': size of the barriers in returns
    """
    # Calculate simple returns
    rets = close.pct_change()
    
    # Calculate volatility using standard deviation
    vol = rets.rolling(window=window).std()
    
    # Get volatility at event times
    event_vol = vol[events.index]
    
    # Calculate target sizes (maximum of volatility-based and minimum threshold)
    pt_target = pd.Series(index=events.index)
    sl_target = pd.Series(index=events.index)
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
    
    events['target'] = pt_target  # Keep original target column with PT target
    events['pt'] = entry_prices * (1 + pt_target)
    events['sl'] = entry_prices * (1 - sl_target)
    
    return events

def ptsl_bbands(events, close, window=20, num_std=2):
    """
    Set barriers based on Bollinger Bands logic.
    Upper/lower bands serve as dynamic PT/SL levels.
    
    Parameters:
        events: DataFrame with event start times
        close: Series of close prices
        window: lookback window for moving average
        num_std: number of standard deviations for bands
    """
    # Calculate Bollinger Bands
    ma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    
    entry_prices = close[events.index]
    events['pt'] = ma[events.index] + (num_std * std[events.index])
    events['sl'] = ma[events.index] - (num_std * std[events.index])
    events['target'] = (events['pt'] - entry_prices) / entry_prices
    
    return events

def ptsl_support_resistance(events, close, high, low, window=20, quantile=0.95):
    """
    Set barriers based on recent support/resistance levels.
    
    Parameters:
        events: DataFrame with event start times
        close: Series of close prices
        high: Series of high prices
        low: Series of low prices
        window: lookback window for level calculation
        quantile: quantile for resistance/support levels
    """
    pt_levels = pd.Series(index=events.index)
    sl_levels = pd.Series(index=events.index)
    
    for idx in events.index:
        lookback_idx = max(0, close.index.get_loc(idx) - window)
        hist_high = high.iloc[lookback_idx:close.index.get_loc(idx)]
        hist_low = low.iloc[lookback_idx:close.index.get_loc(idx)]
        
        # Resistance level for profit-taking
        pt_levels[idx] = np.quantile(hist_high, quantile)
        # Support level for stop-loss
        sl_levels[idx] = np.quantile(hist_low, 1 - quantile)
    
    events['pt'] = pt_levels
    events['sl'] = sl_levels
    events['target'] = (events['pt'] - close[events.index]) / close[events.index]
    
    return events

def ptsl_adaptive(events, close, window=20, base_multiplier=2, momentum_window=10):
    """
    Adaptive barriers that adjust based on recent price momentum.
    Wider barriers in momentum direction, tighter in opposite direction.
    
    Parameters:
        events: DataFrame with event start times
        close: Series of close prices
        window: volatility lookback window
        base_multiplier: base volatility multiplier
        momentum_window: window for momentum calculation
    """
    # Calculate volatility
    vol = close.pct_change().rolling(window=window).std()
    
    # Calculate momentum indicator (e.g., ROC)
    momentum = close.pct_change(momentum_window)
    momentum_std = momentum.rolling(window=window).std()
    momentum_score = momentum / momentum_std
    
    # Adjust multipliers based on momentum
    pt_multiplier = pd.Series(base_multiplier, index=events.index)
    sl_multiplier = pd.Series(base_multiplier, index=events.index)
    
    momentum_at_events = momentum_score[events.index]
    pt_multiplier[momentum_at_events > 0] *= (1 + momentum_at_events[momentum_at_events > 0])
    sl_multiplier[momentum_at_events > 0] *= (1 - momentum_at_events[momentum_at_events > 0] * 0.5)
    pt_multiplier[momentum_at_events < 0] *= (1 - abs(momentum_at_events[momentum_at_events < 0]) * 0.5)
    sl_multiplier[momentum_at_events < 0] *= (1 + abs(momentum_at_events[momentum_at_events < 0]))
    
    # Calculate barriers
    entry_prices = close[events.index]
    events['pt'] = entry_prices * (1 + pt_multiplier * vol[events.index])
    events['sl'] = entry_prices * (1 - sl_multiplier * vol[events.index])
    events['target'] = (events['pt'] - entry_prices) / entry_prices
    
    return events


def ptsl_regime_adaptive(events, close, high, low, volume, 
                        window=20, base_multiplier=2, 
                        regime_window=50, n_regimes=3):
    """
    Advanced barrier setting that adapts to market regimes using multiple features
    and clustering to identify market states.
    
    This function implements a sophisticated approach to setting profit-taking (PT) and 
    stop-loss (SL) barriers by analyzing the current market regime. It:
    
    1. Calculates multiple market features (volatility, trend, volume, price ranges)
       to capture different aspects of market behavior
    
    2. Uses K-means clustering to identify distinct market regimes (e.g., trending,
       ranging, volatile) based on these features
       
    3. Adjusts barrier levels dynamically based on:
       - The identified market regime
       - Mean reversion tendencies (tighter PT when overbought, wider when oversold)
       - Support and resistance levels from recent price history
       - Regime-specific volatility and trend characteristics
    
    This adaptive approach aims to optimize trade exits by being more conservative
    in high-risk regimes and more aggressive in favorable conditions.
    
    Parameters:
        events: DataFrame with event start times
        close, high, low, volume: price and volume Series
        window: base lookback window for calculating features
        base_multiplier: base volatility multiplier for barrier width
        regime_window: window for regime detection (longer = more stable regimes)
        n_regimes: number of market regimes to identify (typically 2-4)
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    # 1. Calculate various market features
    features = pd.DataFrame(index=close.index)
    
    # Volatility features
    features['real_vol'] = np.log(high/low)
    features['close_vol'] = close.pct_change().rolling(window).std()
    
    # Trend features
    features['roc'] = close.pct_change(window)
    features['ma_ratio'] = close / close.rolling(window).mean()
    
    # Volume features
    features['vol_ratio'] = volume / volume.rolling(window).mean()
    
    # Range features
    features['high_low_ratio'] = (high - low) / close
    
    # 2. Identify market regimes using clustering
    scaler = StandardScaler()
    regime_features = features.rolling(regime_window).mean()
    regime_features = regime_features.dropna()
    
    # Fit KMeans on scaled features
    scaled_features = scaler.fit_transform(regime_features)
    kmeans = KMeans(n_clusters=n_regimes, random_state=42)
    regimes = pd.Series(kmeans.fit_predict(scaled_features), 
                       index=regime_features.index)
    
    # 3. Calculate dynamic multipliers based on regime
    pt_multipliers = pd.Series(base_multiplier, index=events.index)
    sl_multipliers = pd.Series(base_multiplier, index=events.index)
    
    for regime in range(n_regimes):
        regime_mask = regimes[events.index] == regime
        
        # Calculate regime-specific volatility
        regime_vol = features.loc[regimes == regime, 'close_vol'].mean()
        
        # Calculate regime-specific trend strength
        trend_strength = abs(features.loc[regimes == regime, 'roc'].mean())
        
        # Adjust multipliers based on regime characteristics
        pt_multipliers[regime_mask] *= (1 + trend_strength)
        sl_multipliers[regime_mask] *= (1 + regime_vol)
    
    # 4. Add mean reversion adjustment
    zscore = (close - close.rolling(window).mean()) / close.rolling(window).std()
    zscore = zscore[events.index]
    
    # Tighten profit targets when extended, widen stops
    pt_multipliers[zscore > 1] *= 0.8
    sl_multipliers[zscore > 1] *= 1.2
    
    # Widen profit targets when oversold, tighten stops
    pt_multipliers[zscore < -1] *= 1.2
    sl_multipliers[zscore < -1] *= 0.8
    
    # 5. Add support/resistance influence
    for idx in events.index:
        lookback_idx = max(0, close.index.get_loc(idx) - window)
        hist_high = high.iloc[lookback_idx:close.index.get_loc(idx)]
        hist_low = low.iloc[lookback_idx:close.index.get_loc(idx)]
        
        # Find nearest support/resistance levels
        entry_price = close[idx]
        resistance_levels = hist_high[hist_high > entry_price]
        support_levels = hist_low[hist_low < entry_price]
        
        if not resistance_levels.empty:
            nearest_resistance = resistance_levels.min()
            pt_distance = (nearest_resistance - entry_price) / entry_price
            pt_multipliers[idx] = min(pt_multipliers[idx], pt_distance)
            
        if not support_levels.empty:
            nearest_support = support_levels.max()
            sl_distance = (entry_price - nearest_support) / entry_price
            sl_multipliers[idx] = min(sl_multipliers[idx], sl_distance)
    
    # 6. Calculate final barriers
    entry_prices = close[events.index]
    vol = features['close_vol'][events.index]
    
    events['pt'] = entry_prices * (1 + pt_multipliers * vol)
    events['sl'] = entry_prices * (1 - sl_multipliers * vol)
    events['target'] = (events['pt'] - entry_prices) / entry_prices
    
    # Store additional metadata for analysis
    events['regime'] = regimes[events.index]
    events['pt_multiplier'] = pt_multipliers
    events['sl_multiplier'] = sl_multipliers
    
    return events
