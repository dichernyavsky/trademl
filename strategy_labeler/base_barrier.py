import pandas as pd
import numpy as np
from numba import njit

# ----- BASE BARRIER CLASS ------------------------------------------------------------------


class Barrier:
    """
    Base class for all barrier strategies.
    Each barrier strategy must implement calculate_barriers method.
    Also handles vertical barriers (t1) and trade generation by default.
    """
    
    def __init__(self, hold_periods=60, **kwargs):
        """    
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
        events = self._calculate_vertical_barrier(events, data, **kwargs)
        
        return events
    
    def _calculate_horizontal_barriers(self, events, data, **kwargs):
        """
        Calculate horizontal barriers (PT/SL).
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _calculate_horizontal_barriers")
    
    def _calculate_vertical_barrier(self, events, data, **kwargs):
        """
        Add vertical barrier (t1) to events.

        Behavior:
        - t1 = min(event_pos + hold_periods, last_index)
        - Uses UniqueBarID for precise synchronization
        """
        hold_periods = kwargs.get('hold_periods', self.hold_periods)

        result = events.copy()
        close_idx = data['Close'].index  # This is UniqueBarID index
        n = len(close_idx)

        # Use UniqueBarID for precise synchronization
        if 'UniqueBarID' in events.columns:
            # Events have UniqueBarID column - use it for exact matching
            event_bar_ids = events['UniqueBarID'].values
            pos = close_idx.get_indexer(event_bar_ids)  # Should always find exact matches
            pos = np.where(pos == -1, n - 1, pos)  # Fallback to last bar if not found
        else:
            # Fallback to index-based approach (for backward compatibility)
            ev_vals = result.index.values
            pos = np.searchsorted(close_idx.values, ev_vals, side='left')
            pos = np.clip(pos, 0, n - 1)
        
        target_pos = np.minimum(pos + int(hold_periods), n - 1)

        # Assign t1 as UniqueBarID (not time)
        result['t1'] = close_idx.values[target_pos]
        return result
    
    def generate_trades(self, events, data, use_hl=True, entry_price_mode: str = 'close', **kwargs):
        """
        Generate trades with fixed profit-take (PT) and stop-loss (SL) barriers
        using a fast path: vectorized index mapping + a Numba window scanner.

        Parameters
        ----------
        events : pd.DataFrame
            Event table indexed by entry timestamps. Must contain:
            - 'direction' (int): +1 for long, -1 for short
            - 'pt' (float): profit-take level per event
            - 'sl' (float): stop-loss level per event
            - 't1' (datetime-like): vertical barrier (time horizon)
            May optionally contain:
            - 'entry_price' (float): if NaN, it will be filled only when the
                exact event timestamp exists in the price index (preserving
                the original behavior).
        data : pd.DataFrame or pd.Series
            Price data. If a DataFrame, must include columns 'High', 'Low', 'Close'
            with a monotonic DateTimeIndex (OHLCV mode). If a Series, it is treated
            as a Close series (Close-only mode).
        use_hl : bool, default True
            If True and OHLCV data is provided, barrier touches are tested against
            High/Low (upper via High >= PT, lower via Low <= SL). Otherwise, tests
            are performed on Close.
        entry_price_mode : str, default 'close'
            How to set entry_price when it's NaN:
            - 'close': use Close price at the actual execution bar (start_pos)
            - 'open': use Open price at the actual execution bar (start_pos)
            - 'exact': only fill if event timestamp exactly matches price index (original behavior)
        **kwargs
            Optional arguments. Recognized key:
            - 'hold_periods' (int): accepted for API compatibility but not used
                here; the vertical barrier is taken from 't1' in `events`.

        Returns
        -------
        pd.DataFrame
            A copy of `events` augmented with:
            - 'bin' (int): outcome label
                    +1 → PT hit for long / SL hit for short (upper touched first)
                    -1 → SL hit for long / PT hit for short (lower touched first)
                    0 → time exit at 't1' (no barrier hit)
            - 'exit_time' (pd.Timestamp): exit timestamp (bar where the hit occurred,
                    or the bar at 't1' for time exits)
            - 'exit_price' (float): exit price (barrier level for PT/SL, Close at 't1'
                    for time exits)
            - 'r_target' (float): direction-adjusted return
                    (exit_price - entry_price) / entry_price * direction
            - 'entry_price' (float): preserved or filled as described above
            Trades that exit on the same bar as they entered are removed.

        Behavior
        --------
        - The price window per event is inclusive: [entry_time … t1].
        Internally, `entry_time` maps to the first bar with time >= event index,
        and `t1` maps to the last bar with time <= 't1'.
        - Barriers are pre-adjusted by direction:
            long  → upper=pt, lower=sl
            short → upper=sl, lower=pt
        - If both barriers are crossed within the same bar when using High/Low,
        the upper barrier is prioritized because it is checked first.
        - Time exit (bin==0) uses Close at the exact 't1' timestamp (assuming 't1'
        was derived from the price index; otherwise it falls back to the last bar
        before or at 't1').
        - 'entry_price' is only auto-filled when the event timestamp matches a price
        index value exactly.
        """
        out = events.copy()
        # Ensure columns exist
        for col, default in [('bin', 0), ('exit_time', pd.NaT), ('exit_price', np.nan),
                            ('r_target', np.nan), ('entry_price', np.nan)]:
            if col not in out.columns:
                out[col] = default

        # Detect OHLCV vs Close series
        is_ohlcv = isinstance(data, pd.DataFrame) and {'High', 'Low', 'Close'}.issubset(data.columns)
        if not is_ohlcv:
            # Accept a Series of Close prices
            close = data.astype(float)
            idx   = close.index.values
            high = low = None
        else:
            high  = data['High'].to_numpy(dtype=np.float64)
            low   = data['Low'].to_numpy(dtype=np.float64)
            close = data['Close'].astype(float)
            idx   = close.index.values

        close_np = close.to_numpy(dtype=np.float64)
        n = len(idx)

        # Vectorized mapping:
        # window [start_pos, end_pos], inclusive.
        # Use UniqueBarID for precise synchronization
        if 'UniqueBarID' in out.columns:
            # Use UniqueBarID for exact matching
            event_bar_ids = out['UniqueBarID'].values
            t1_bar_ids = out['t1'].values  # t1 is now also UniqueBarID
            
            # Direct position mapping using UniqueBarID
            start_pos = pd.Index(close.index).get_indexer(event_bar_ids)
            end_pos_r = pd.Index(close.index).get_indexer(t1_bar_ids)
        else:
            # Fallback to time-based approach
            ev_times = out.index.values
            t1_times = out['t1'].values

            # searchsorted on numpy datetime64
            start_pos = np.searchsorted(idx, ev_times, side='left')
            end_pos_r = np.searchsorted(idx, t1_times, side='right') - 1

        # Clip invalid positions
        start_pos = np.where(start_pos < 0, 0, start_pos)
        start_pos = np.where(start_pos >= n, n - 1, start_pos)
        end_pos_r = np.where(end_pos_r < 0, -1, end_pos_r)      # -1 -> invalid (time exit will handle)
        end_pos_r = np.where(end_pos_r >= n, n - 1, end_pos_r)

        # Prepare per-event barriers adjusted by direction
        direction = out['direction'].astype(int).to_numpy()
        pt = out['pt'].astype(float).to_numpy()
        sl = out['sl'].astype(float).to_numpy()

        upper = np.where(direction == 1, pt, sl).astype(np.float64)
        lower = np.where(direction == 1, sl, pt).astype(np.float64)

        # Fill entry_price for events where it's NaN
        need_fill = out['entry_price'].isna().to_numpy()
        if need_fill.any():
            if entry_price_mode == 'exact':
                # Use UniqueBarID for exact matching if available
                if 'UniqueBarID' in out.columns:
                    event_bar_ids = out.loc[need_fill, 'UniqueBarID'].values
                    exact_pos = pd.Index(close.index).get_indexer(event_bar_ids)  # Should always find exact matches
                    mask_exact = (exact_pos >= 0) & need_fill
                    if mask_exact.any():
                        out.loc[mask_exact, 'entry_price'] = close.iloc[exact_pos[mask_exact]].to_numpy()
                else:
                    # Fallback: only fill if event timestamp exactly matches price index
                    exact_pos = pd.Index(close.index).get_indexer(out.index)  # -1 if not exact
                    mask_exact = (exact_pos >= 0) & need_fill
                    if mask_exact.any():
                        out.loc[mask_exact, 'entry_price'] = close.iloc[exact_pos[mask_exact]].to_numpy()
            else:
                # New behavior: use price at the actual execution bar (start_pos)
                sp = np.clip(start_pos[need_fill], 0, n-1)
                if isinstance(data, pd.DataFrame) and entry_price_mode == 'open' and 'Open' in data.columns:
                    prices = data['Open'].to_numpy(dtype=np.float64)
                else:
                    prices = close_np
                out.loc[need_fill, 'entry_price'] = prices[sp]

        # Run Numba scanner
        if not is_ohlcv:
            high_np = low_np = close_np  # unused when is_ohlcv=False
        else:
            high_np = high
            low_np  = low

        exit_pos, exit_price, bin_arr = _scan_barriers_numba(
            start_pos.astype(np.int64),
            end_pos_r.astype(np.int64),
            high_np, low_np, close_np,
            upper, lower,
            bool(use_hl), bool(is_ohlcv),
            direction.astype(np.int8)
        )

        # Assign exits back to DataFrame
        # Exit time from position (time-exit already encoded as end_pos in numba)
        if is_ohlcv and 'OpenTime' in data.columns:
            exit_time = data['OpenTime'].iloc[np.clip(exit_pos, 0, n-1)].values
        else:
            exit_time = pd.to_datetime(idx[np.clip(exit_pos, 0, n-1)])
        out['exit_time']  = exit_time
        out['exit_price'] = exit_price
        out['bin']        = bin_arr.astype(int)

        # Time exit (bin==0) should use close at barrier_time (t1)
        time_mask = (out['bin'] == 0)
        if time_mask.any():
            # end_pos_r already points to last bar <= t1, we want the bar at t1 exactly
            # Your original code used close at barrier_time (t1). Since t1 is aligned, fetch exact:
            t1_exact_pos = pd.Index(close.index).get_indexer(out.loc[time_mask, 't1'].values)
            # Fallback if t1 not found (shouldn't happen if your t1 was built from data index)
            t1_exact_pos = np.where(t1_exact_pos >= 0, t1_exact_pos, end_pos_r[time_mask.values])
            out.loc[time_mask, 'exit_price'] = close_np[t1_exact_pos]
            
            # Get exit time from OpenTime column if available, otherwise use index
            if is_ohlcv and 'OpenTime' in data.columns:
                out.loc[time_mask, 'exit_time'] = data['OpenTime'].iloc[t1_exact_pos].values
            else:
                out.loc[time_mask, 'exit_time'] = pd.to_datetime(idx[t1_exact_pos])

        # Compute r_target when entry/exit are available
        ep = out['entry_price'].to_numpy(dtype=np.float64)
        xp = out['exit_price'].to_numpy(dtype=np.float64)
        good = (~np.isnan(ep)) & (~np.isnan(xp)) & (ep != 0.0)
        if good.any():
            ret = (xp[good] - ep[good]) / ep[good]
            out.loc[good, 'r_target'] = ret * out.loc[good, 'direction'].to_numpy()

        # Drop trades that exit on the same bar as they entered (as in your original)
        same_bar_mask = out['exit_time'].notna() & (out['exit_time'] == out['OpenTime'])
        out = out.loc[~same_bar_mask].copy()

        return out
    
    def validate_config(self):
        """
        Validate barrier configuration.
        Override in subclasses if needed.
        """
        return True
    
# ---------------------- Helper function --------------------------------
    
@njit
def _scan_barriers_numba(start_pos, end_pos, high, low, close,
                        upper, lower, use_hl, is_ohlcv, direction):
    """
    Scan per-event windows to find the first barrier touch.

    For each event i, this scans the inclusive window [start_pos[i], end_pos[i]] over the
    price arrays and determines whether the UPPER (pt) or LOWER (sl) barrier is hit first.
    If neither barrier is hit within the window, the exit falls back to a time exit at
    end_pos[i]. The function supports either OHLCV (using High/Low) or Close-only paths.

    Args:
        start_pos (np.ndarray[int64], shape (m,)):
            Start index (inclusive) in the price arrays for each event.
        end_pos   (np.ndarray[int64], shape (m,)):
            End index (inclusive) in the price arrays for each event.
            Values < 0 indicate an invalid window and will yield a time exit with NaNs.
        high (np.ndarray[float64], shape (n,)):
            High prices. Used only when `is_ohlcv=True` and `use_hl=True`.
        low  (np.ndarray[float64], shape (n,)):
            Low prices. Used only when `is_ohlcv=True` and `use_hl=True`.
        close (np.ndarray[float64], shape (n,)):
            Close prices. Always required. Used for close-path scanning and time exits.
        upper (np.ndarray[float64], shape (m,)):
            Upper (profit-take) barrier level per event (already adjusted for direction).
        lower (np.ndarray[float64], shape (m,)):
            Lower (stop-loss) barrier level per event (already adjusted for direction).
        use_hl (bool):
            If True and `is_ohlcv` is True, test barrier touches using High/Low.
            Otherwise, test using Close.
        is_ohlcv (bool):
            If True, `high`/`low` are considered available and may be used with `use_hl`.
        direction (np.ndarray[int8], shape (m,)):
            Trade direction per event: +1 for long, -1 for short.

    Returns:
        exit_pos (np.ndarray[int64], shape (m,)):
            Exit index in the price arrays for each event.
            If no window or invalid end index, may be -1.
        exit_price (np.ndarray[float64], shape (m,)):
            Exit price: barrier level when a barrier is hit; otherwise Close[end_pos].
            May be NaN if end_pos < 0.
        bin_arr (np.ndarray[int8], shape (m,)):
            Outcome label per event:
              - +1: profit-take for a long OR stop-loss for a short (upper hit with dir=+1,
                    or lower hit with dir=-1)
              - -1: stop-loss for a long OR profit-take for a short (lower hit with dir=+1,
                    or upper hit with dir=-1)
              -  0: time exit (no barrier hit within the window)

    Notes:
        - Windows are inclusive on both ends.
        - If both barriers are crossed on the same bar (e.g., High >= upper and Low <= lower),
          the function prioritizes the upper barrier because it is checked first.
        - NaN in `upper` or `lower` disables barrier checking for that event and
          results in a time exit.
        - Inputs should be 1D contiguous NumPy arrays with consistent indexing.
    """
    m = len(start_pos)
    exit_pos   = np.empty(m, np.int64)
    exit_price = np.empty(m, np.float64)
    bin_arr    = np.zeros(m, np.int8)  # 1 / -1 / 0

    for i in range(m):
        s = start_pos[i]
        e = end_pos[i]
        # Default is time exit at e (if window invalid, also behaves as time exit)
        ep = e
        price = close[e] if e >= 0 else np.nan
        hit = 0  # 1 -> upper/PT, -1 -> lower/SL, 0 -> none (time exit)

        if s >= 0 and e >= 0 and s <= e and not np.isnan(upper[i]) and not np.isnan(lower[i]):
            for j in range(s, e + 1):
                if is_ohlcv and use_hl:
                    if high[j] >= upper[i]:
                        hit = 1
                        ep = j
                        price = upper[i]
                        break
                    if low[j] <= lower[i]:
                        hit = -1
                        ep = j
                        price = lower[i]
                        break
                else:
                    c = close[j]
                    if c >= upper[i]:
                        hit = 1
                        ep = j
                        price = upper[i]
                        break
                    if c <= lower[i]:
                        hit = -1
                        ep = j
                        price = lower[i]
                        break

        # Map hit + direction to bin label
        if hit == 1:
            bin_val = 1 if direction[i] == 1 else -1
        elif hit == -1:
            bin_val = -1 if direction[i] == 1 else 1
        else:
            bin_val = 0

        exit_pos[i]   = ep
        exit_price[i] = price
        bin_arr[i]    = bin_val

    return exit_pos, exit_price, bin_arr