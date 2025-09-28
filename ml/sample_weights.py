"""
Sample Weights Module for ML

This module provides functionality to calculate sample weights for machine learning models.
Sample weights are used to give different importance to different training samples based on
various criteria such as:
- Trade performance (returns, Sharpe ratio, etc.)
- Market conditions
- Volatility regimes
- Time decay
- Custom business logic

The weights are used during model training to optimize for specific objectives.
"""

import numpy as np
import pandas as pd


def num_co_events_from_intervals(events: pd.DataFrame,
                                 t1_col: str = 't1',
                                 fallback_freq: str = '1min') -> tuple[pd.Series, pd.DatetimeIndex]:
    """Возвращает:
       ct: число совместно активных событий на внутренней временной сетке
       grid: внутренняя временная сетка (DatetimeIndex)
    """
    t0 = pd.to_datetime(events.index)
    t1 = pd.to_datetime(events[t1_col], errors='coerce')
    ok = t0.notna() & t1.notna() & (t1.values >= t0.values)
    t0, t1 = t0[ok], t1[ok]
    if len(t0) == 0:
        return pd.Series(dtype='int64'), pd.DatetimeIndex([])

    # частота сетки
    freq = (pd.infer_freq(t0.sort_values()) or
            pd.infer_freq(t1.sort_values()) or
            fallback_freq)
    grid = pd.date_range(start=t0.min(), end=t1.max(), freq=freq)

    # delta-алгоритм
    t0_pos = np.searchsorted(grid.values, t0.values, side='left')
    t1_pos = np.searchsorted(grid.values, t1.values, side='right') - 1
    valid = (t0_pos >= 0) & (t0_pos < len(grid)) & (t1_pos >= t0_pos) & (t1_pos < len(grid))

    deltas = np.zeros(len(grid), dtype=np.int64)
    np.add.at(deltas, t0_pos[valid], 1)
    end_pos = t1_pos[valid] + 1
    mask_end = end_pos < len(grid)
    np.add.at(deltas, end_pos[mask_end], -1)

    ct = pd.Series(deltas.cumsum(), index=grid, name='numCoEvents')
    return ct, grid


def avg_uniqueness(events: pd.DataFrame,
                   ct: pd.Series,
                   t1_col: str = 't1') -> pd.Series:
    """ubar_i = mean_{t in [t0_i, t1_i]} 1 / c_t"""
    t0 = pd.to_datetime(events.index)
    t1 = pd.to_datetime(events[t1_col], errors='coerce')
    ubar = pd.Series(index=events.index, dtype='float64')
    for i, (ti, to) in enumerate(zip(t0, t1)):
        if pd.isna(ti) or pd.isna(to) or to < ti:
            ubar.iloc[i] = 0.0
            continue
        denom = ct.loc[ti:to].astype(float).values
        if len(denom) == 0:
            ubar.iloc[i] = 0.0
        else:
            ubar.iloc[i] = float((1.0 / np.maximum(denom, 1.0)).mean())
    return ubar.fillna(0.0)


def return_weights(close: pd.Series,
                   events: pd.DataFrame,
                   ct: pd.Series,
                   t1_col: str = 't1',
                   use_abs: bool = True,
                   normalize: bool = True) -> pd.Series:
    """
    w_ret_i = sum_{t in [t0_i, t1_i]} (|r_t| or r_t) / c_t ; по умолчанию берём |r|
    затем опциональная нормировка sum w = I (I = число событий)
    """
    ret = np.log(close).diff()
    w = pd.Series(index=events.index, dtype='float64')
    for tIn, tOut in zip(events.index, pd.to_datetime(events[t1_col], errors='coerce')):
        if pd.isna(tOut) or tOut < tIn:
            w.loc[tIn] = 0.0
            continue
        seg = ret.loc[tIn:tOut]
        den = ct.loc[tIn:tOut].astype(float).replace(0, np.nan)
        num = seg.abs() if use_abs else seg
        w.loc[tIn] = (num.div(den)).sum()
    w = w.fillna(0.0)
    if normalize and w.sum() > 0:
        w *= (len(w) / w.sum())
    return w


def time_decay_from_ubar(ubar: pd.Series, clf_last_w: float = 1.0) -> pd.Series:
    """Монотонная линейная функция от cumulative sum(ubar). Последнее значение = clf_last_w."""
    cs = ubar.sort_index().cumsum()
    if len(cs) == 0 or cs.iloc[-1] == 0:
        return pd.Series(1.0, index=ubar.index)
    if clf_last_w >= 0:
        slope = (1.0 - clf_last_w) / cs.iloc[-1]
    else:
        slope = 1.0 / ((clf_last_w + 1.0) * cs.iloc[-1])
    const = 1.0 - slope * cs.iloc[-1]
    out = const + slope * cs
    out[out < 0] = 0.0
    return out.reindex(ubar.index)


def build_event_weights(close: pd.Series,
                        events: pd.DataFrame,
                        t1_col: str = 't1',
                        fallback_freq: str = '1min',
                        clf_last_w: float = 1.0,
                        use_abs_ret: bool = True,
                        normalize_ret: bool = True) -> pd.DataFrame:
    """
    Возвращает DataFrame с колонками:
      ['w_uniqueness', 'w_return', 'w_time', 'w_event']
    где w_event = w_uniqueness * w_return * w_time  (без финальной нормировки)
    """
    ct, _ = num_co_events_from_intervals(events, t1_col=t1_col, fallback_freq=fallback_freq)
    ubar = avg_uniqueness(events, ct, t1_col=t1_col)
    w_ret = return_weights(close, events, ct, t1_col=t1_col, use_abs=use_abs_ret, normalize=normalize_ret)
    w_time = time_decay_from_ubar(ubar, clf_last_w=clf_last_w)

    out = pd.DataFrame({
        'w_uniqueness': ubar.reindex(events.index).fillna(0.0),
        'w_return':     w_ret.reindex(events.index).fillna(0.0),
        'w_time':       w_time.reindex(events.index).fillna(0.0),
    }, index=events.index)
    out['w_event'] = (out['w_uniqueness'] * out['w_return'] * out['w_time']).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out


def get_sample_weights_for_ml(events: pd.DataFrame, 
                             close: pd.Series,
                             weight_type: str = 'w_event',
                             t1_col: str = 't1',
                             **kwargs) -> np.ndarray:
    """
    Calculate sample weights for ML training.
    
    Args:
        events: DataFrame with events (trades)
        close: Series with close prices
        weight_type: Type of weight to use ('w_event', 'w_return', 'w_uniqueness', 'w_time')
        t1_col: Column name for exit times
        **kwargs: Additional parameters for build_event_weights
        
    Returns:
        np.ndarray: Sample weights for ML training
    """
    # Calculate all weights
    weights_df = build_event_weights(close, events, t1_col=t1_col, **kwargs)
    
    # Return the requested weight type
    if weight_type not in weights_df.columns:
        raise ValueError(f"Weight type '{weight_type}' not found. Available: {list(weights_df.columns)}")
    
    return weights_df[weight_type].values 