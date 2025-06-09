import numpy as np
import pandas as pd

def process_event(price_data, upper_barrier, trailing_pct, save_trail, use_hl=False):
    """
    Vectorized calculation of trailing stop or profit-taking triggers.
    
    Parameters:
      price_data: pandas Series (close) or DataFrame (OHLCV) with prices from entry to vertical barrier
      upper_barrier: fixed profit-taking level
      trailing_pct: trailing amount (e.g., 0.05 for 5%)
      save_trail: if True, returns the series of dynamic stop values
      use_hl: if True and price_data is DataFrame, use high/low for barrier touches
      
    Returns:
      triggered: tuple (bin, exit_time) or (None, None) if no barrier triggered;
                 bin == 1 - profit-taking triggered, bin == -1 - trailing-stop triggered
      trail_series: pandas Series with dynamic stop values (if save_trail True), else None
    """
    # Check if we have a DataFrame with OHLCV data or just a Series with close prices
    is_ohlcv = isinstance(price_data, pd.DataFrame) and 'High' in price_data.columns and 'Low' in price_data.columns
    
    times = price_data.index
    
    # Extract the relevant price series based on the data type and use_hl parameter
    if is_ohlcv and use_hl:
        # For checking profit take, use high prices
        prices_high = price_data['High'].values
        # For checking stop loss, use low prices
        prices_low = price_data['Low'].values
        # For trailing calculation, use close prices
        prices_close = price_data['Close'].values
    else:
        # If we only have close prices or use_hl is False, use close for everything
        prices = price_data.values if isinstance(price_data, pd.Series) else price_data['Close'].values
        prices_high = prices
        prices_low = prices
        prices_close = prices

    # Calculate cumulative maximum of close prices for trailing stop
    cum_max = np.maximum.accumulate(prices_close)
    # Calculate dynamic stop: maximum * (1 - trailing_pct)
    dynamic_sl = cum_max * (1 - trailing_pct)

    # Condition for profit-taking: high price >= upper_barrier
    profit_indices = np.where(prices_high >= upper_barrier)[0]
    # Condition for trailing stop trigger: low price <= dynamic stop
    stop_indices = np.where(prices_low <= dynamic_sl)[0]

    # Initialize variables
    profit_idx = profit_indices[0] if profit_indices.size > 0 else None
    stop_idx = stop_indices[0] if stop_indices.size > 0 else None

    # Determine which event occurred first, if both triggered
    if profit_idx is None and stop_idx is None:
        triggered = (0, None)
    elif profit_idx is None:
        triggered = (-1, times[stop_idx])
    elif stop_idx is None:
        triggered = (1, times[profit_idx])
    else:
        # Compare timestamps
        if profit_idx <= stop_idx:
            triggered = (1, times[profit_idx])
        else:
            triggered = (-1, times[stop_idx])
    
    # Create trail_series if needed
    trail_series = pd.Series(dynamic_sl, index=times) if save_trail else None

    return triggered, trail_series


def generate_trades(price_data, events, trailing_stop=False, trailing_pct=0.0, save_trail=False, use_hl=False):
    """
    Generate trade outcomes based on price data and event definitions.
    
    Parameters:
      price_data: pandas Series (close) or DataFrame (OHLCV) with price data
      events: DataFrame with trade entry points and barrier definitions
      trailing_stop: if True, use trailing stop instead of fixed stop
      trailing_pct: trailing percentage amount
      save_trail: if True, save the dynamic stop path
      use_hl: if True and price_data is DataFrame, use high/low for barrier touches
      
    Returns:
      DataFrame with trade outcomes
    """
    out = events.copy()
    out['bin'] = 0  # default label
    if save_trail:
        out['dynamic_stop_path'] = None  # storage for stop dynamics
    
    # Add columns for exit information
    out['exit_time'] = pd.NaT
    out['exit_price'] = np.nan
    
    # Add entry price column if not already present
    if 'entry_price' not in out.columns:
        out['entry_price'] = np.nan

    # Check if we have a DataFrame with OHLCV data or just a Series with close prices
    is_ohlcv = isinstance(price_data, pd.DataFrame) and 'High' in price_data.columns and 'Low' in price_data.columns
    
    # For accessing close prices later
    close_data = price_data['Close'] if is_ohlcv else price_data

    for loc, barrier_time in out['t1'].items():
        # Set entry price if not already set (do this for each trade in the same loop)
        if pd.isna(out.at[loc, 'entry_price']) and loc in close_data.index:
            out.at[loc, 'entry_price'] = close_data.loc[loc]
            
        # Get price path from entry to barrier
        if is_ohlcv:
            price_path = price_data.loc[loc:barrier_time]
        else:
            price_path = price_data.loc[loc:barrier_time]
            
        if len(price_path) == 0:
            continue

        upper_barrier = out.at[loc, 'pt']
        fixed_sl = out.at[loc, 'sl']

        if trailing_stop and trailing_pct > 0:
            triggered, trail_series = process_event(price_path, upper_barrier, trailing_pct, save_trail, use_hl)
            out.at[loc, 'bin'] = triggered[0]
            out.at[loc, 'exit_time'] = triggered[1]
            
            # Set exit price based on which barrier was hit
            if triggered[0] == 1:  # Profit target hit
                out.at[loc, 'exit_price'] = upper_barrier
            elif triggered[0] == -1:  # Trailing stop hit
                # Get the dynamic stop level at the exit time
                if triggered[1] is not None:
                    stop_idx = list(price_path.index).index(triggered[1])
                    dynamic_sl_value = trail_series.iloc[stop_idx]
                    out.at[loc, 'exit_price'] = dynamic_sl_value
            elif triggered[1] is not None:  # Vertical barrier (time exit)
                out.at[loc, 'exit_price'] = close_data.loc[triggered[1]]
                
            if save_trail:
                out.at[loc, 'dynamic_stop_path'] = trail_series
        else:
            if is_ohlcv and use_hl:
                # For fixed stop, use high/low prices for checking barriers
                touched_pt = price_path['High'][price_path['High'] >= upper_barrier]
                touched_sl = price_path['Low'][price_path['Low'] <= fixed_sl]
            else:
                # Use close prices
                close_path = price_path['Close'] if is_ohlcv else price_path
                touched_pt = close_path[close_path >= upper_barrier]
                touched_sl = close_path[close_path <= fixed_sl]

            if not touched_pt.empty and (touched_sl.empty or touched_pt.index[0] <= touched_sl.index[0]):
                out.at[loc, 'bin'] = 1
                out.at[loc, 'exit_time'] = touched_pt.index[0]
                out.at[loc, 'exit_price'] = upper_barrier  # Use PT level as exit price
            elif not touched_sl.empty and (touched_pt.empty or touched_sl.index[0] < touched_pt.index[0]):
                out.at[loc, 'bin'] = -1
                out.at[loc, 'exit_time'] = touched_sl.index[0]
                out.at[loc, 'exit_price'] = fixed_sl  # Use SL level as exit price
            else:
                out.at[loc, 'bin'] = 0
                # If no barrier triggered, set exit_time to vertical barrier (t1)
                out.at[loc, 'exit_time'] = barrier_time
                out.at[loc, 'exit_price'] = close_data.loc[barrier_time]

    return out
