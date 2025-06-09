import pandas as pd
import numpy as np

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
    entry_prices = close[events.index]
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
