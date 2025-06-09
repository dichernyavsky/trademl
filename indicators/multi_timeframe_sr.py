import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
from .base_indicator import BaseIndicator

class MultiTimeframeSR(BaseIndicator):
    """
    Multi-timeframe support and resistance indicator.
    
    This indicator identifies support and resistance levels across multiple timeframes
    and classifies them as major or minor based on their strength and significance.
    """
    
    def __init__(self, timeframes=None, lookback_periods=None, volatility_factor=2.0,
                 cluster_eps=0.005, min_samples=2, major_threshold=5.0,
                 volume_weight=1.0, timeframe_weights=None):
        """
        Initialize the multi-timeframe support and resistance indicator.
        
        Args:
            timeframes (list): List of timeframes to analyze, e.g., ['1D', '4H', '1H']
                               If None, will use the provided data as is
            lookback_periods (dict): Dict of lookback periods for each timeframe
                               e.g., {'1D': 100, '4H': 200, '1H': 400}
            volatility_factor (float): Factor to multiply ATR for adaptive pivot window
            cluster_eps (float): Maximum distance between points in a cluster (as % of price)
            min_samples (int): Minimum number of points to form a cluster
            major_threshold (float): Threshold score for classifying a level as major
            volume_weight (float): Weight of volume in the level strength calculation
            timeframe_weights (dict): Weights for each timeframe,
                               e.g., {'1D': 3.0, '4H': 2.0, '1H': 1.0}
        """
        super().__init__(name="Multi-Timeframe S/R")
        
        # Set default timeframes if not provided
        self.timeframes = timeframes or ['1D', '4H', '1H']
        
        # Set default lookback periods if not provided
        self.lookback_periods = lookback_periods or {
            '1D': 100,  # 100 days
            '4H': 150,  # 150 4-hour periods (~25 days)
            '1H': 240   # 240 hours (~10 days)
        }
        
        # Set default timeframe weights if not provided
        self.timeframe_weights = timeframe_weights or {
            '1D': 3.0,  # Daily timeframe has highest weight
            '4H': 2.0,  # 4-hour timeframe has medium weight
            '1H': 1.0   # Hourly timeframe has lowest weight
        }
        
        self.volatility_factor = volatility_factor
        self.cluster_eps = cluster_eps
        self.min_samples = min_samples
        self.major_threshold = major_threshold
        self.volume_weight = volume_weight
        
        # Storage for calculated extrema and levels
        self.extrema = {}
        self.all_extrema = []
        self.level_scores = {}
        self.is_calculated = False
    
    def calculate(self, data, resample=True):
        """
        Calculate support and resistance levels across multiple timeframes.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            resample (bool): Whether to resample data to different timeframes
                           If False, assumes data is already at the lowest timeframe
        
        Returns:
            dict: Dictionary with major and minor support/resistance levels and their scores
        """
        # Reset storage
        self.extrema = {}
        self.all_extrema = []
        self.level_scores = {}
        
        # Check if we have the necessary data
        required_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        # Process each timeframe
        if resample and self.timeframes is not None:
            # Resample data to different timeframes
            for tf in self.timeframes:
                resampled_data = self._resample_data(data, tf)
                lookback = self.lookback_periods.get(tf, 100)
                
                # Use only the lookback period
                if len(resampled_data) > lookback:
                    resampled_data = resampled_data.iloc[-lookback:]
                
                # Calculate ATR for adaptive window
                atr = self._calculate_atr(resampled_data, period=14)
                
                # Find extrema with adaptive window
                self.extrema[tf] = self._find_extrema(resampled_data, atr, tf)
                
                # Add to all extrema list with timeframe info
                for ext_type, points in self.extrema[tf].items():
                    for point in points:
                        self.all_extrema.append({
                            'timestamp': point['timestamp'],
                            'price': point['price'],
                            'type': ext_type,  # 'high' or 'low'
                            'timeframe': tf,
                            'volume': point.get('volume', 1.0)
                        })
        else:
            # Use the data as is (assuming it's already at the correct timeframe)
            # Calculate ATR for adaptive window
            atr = self._calculate_atr(data, period=14)
            
            # Find extrema with adaptive window
            tf = self.timeframes[0] if self.timeframes else '1H'  # Default timeframe
            self.extrema[tf] = self._find_extrema(data, atr, tf)
            
            # Add to all extrema list with timeframe info
            for ext_type, points in self.extrema[tf].items():
                for point in points:
                    self.all_extrema.append({
                        'timestamp': point['timestamp'],
                        'price': point['price'],
                        'type': ext_type,  # 'high' or 'low'
                        'timeframe': tf,
                        'volume': point.get('volume', 1.0)
                    })
        
        # Cluster extrema to form zones
        highs, lows = self._cluster_extrema()
        
        # Calculate level scores
        high_scores = self._calculate_level_scores(highs, data)
        low_scores = self._calculate_level_scores(lows, data)
        
        # Classify levels as major or minor
        major_resistance = [level for level, score in high_scores.items() if score >= self.major_threshold]
        minor_resistance = [level for level, score in high_scores.items() if score < self.major_threshold]
        
        major_support = [level for level, score in low_scores.items() if score >= self.major_threshold]
        minor_support = [level for level, score in low_scores.items() if score < self.major_threshold]
        
        # Sort levels by price
        major_resistance.sort(reverse=True)
        minor_resistance.sort(reverse=True)
        major_support.sort()
        minor_support.sort()
        
        # Store results
        self.values = {
            'major_resistance': major_resistance,
            'minor_resistance': minor_resistance,
            'major_support': major_support,
            'minor_support': minor_support,
            'resistance_scores': {lvl: high_scores[lvl] for lvl in major_resistance + minor_resistance},
            'support_scores': {lvl: low_scores[lvl] for lvl in major_support + minor_support}
        }
        
        self.is_calculated = True
        
        return self.values
    
    def _resample_data(self, data, timeframe):
        """
        Resample data to the specified timeframe.
        
        Args:
            data (pd.DataFrame): OHLCV data
            timeframe (str): Timeframe to resample to, e.g., '1D', '4H', '1H'
            
        Returns:
            pd.DataFrame: Resampled data
        """
        # Ensure data has datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a DatetimeIndex for resampling")
        
        # Resample data
        ohlc_dict = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }
        
        # Handle volume if available
        if 'Volume' in data.columns:
            ohlc_dict['Volume'] = 'sum'
        
        # Resample
        resampled = data.resample(timeframe).agg(ohlc_dict)
        
        # Drop any rows with NaN values
        resampled = resampled.dropna()
        
        return resampled
    
    def _calculate_atr(self, data, period=14):
        """
        Calculate Average True Range (ATR) for adaptive window sizing.
        
        Args:
            data (pd.DataFrame): OHLCV data
            period (int): ATR period
            
        Returns:
            np.ndarray: ATR values
        """
        high = data['High'].values
        low = data['Low'].values
        close = np.roll(data['Close'].values, 1)
        close[0] = data['Open'].values[0]  # Use open for the first bar
        
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - close)
        tr3 = np.abs(low - close)
        
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        
        # Calculate ATR using simple moving average
        atr = np.zeros_like(tr)
        atr[:period] = np.mean(tr[:period])
        
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        return atr
    
    def _find_extrema(self, data, atr, timeframe):
        """
        Find local extrema (highs and lows) using adaptive window based on volatility.
        
        Args:
            data (pd.DataFrame): OHLCV data
            atr (np.ndarray): ATR values
            timeframe (str): Current timeframe
            
        Returns:
            dict: Dictionary with high and low extrema
        """
        extrema = {'high': [], 'low': []}
        
        high = data['High'].values
        low = data['Low'].values
        volume = data.get('Volume', pd.Series(np.ones(len(data)))).values
        
        # Default pivot lookback and lookahead
        default_window = 2
        
        for i in range(default_window, len(data) - default_window):
            # Get current ATR
            current_atr = atr[i]
            
            # Calculate adaptive window size based on volatility
            # Higher volatility -> larger window
            window_size = max(default_window, int(current_atr * self.volatility_factor / (data['Close'].iloc[i] * 0.001)))
            
            # Adjust window if we're near the edges
            left_window = min(window_size, i)
            right_window = min(window_size, len(data) - 1 - i)
            
            # Check if we have a pivot high
            if high[i] > np.max(high[i-left_window:i]) and high[i] > np.max(high[i+1:i+right_window+1]):
                extrema['high'].append({
                    'timestamp': data.index[i],
                    'price': high[i],
                    'volume': volume[i],
                    'timeframe': timeframe
                })
            
            # Check if we have a pivot low
            if low[i] < np.min(low[i-left_window:i]) and low[i] < np.min(low[i+1:i+right_window+1]):
                extrema['low'].append({
                    'timestamp': data.index[i],
                    'price': low[i],
                    'volume': volume[i],
                    'timeframe': timeframe
                })
        
        return extrema
    
    def _cluster_extrema(self):
        """
        Cluster extrema to form zones of support and resistance.
        
        Returns:
            tuple: (clustered_highs, clustered_lows) - dictionaries of clustered price levels
        """
        if not self.all_extrema:
            return {}, {}
        
        # Extract prices from extrema
        high_prices = np.array([ext['price'] for ext in self.all_extrema if ext['type'] == 'high']).reshape(-1, 1)
        low_prices = np.array([ext['price'] for ext in self.all_extrema if ext['type'] == 'low']).reshape(-1, 1)
        
        # Corresponding extrema
        high_extrema = [ext for ext in self.all_extrema if ext['type'] == 'high']
        low_extrema = [ext for ext in self.all_extrema if ext['type'] == 'low']
        
        # Cluster high prices if we have enough points
        clustered_highs = {}
        if len(high_prices) >= self.min_samples:
            # Scale eps by price level for percentage-based clustering
            if len(high_prices) > 0:
                mean_price = np.mean(high_prices)
                high_eps = mean_price * self.cluster_eps
                
                # Cluster high extrema
                high_clusters = DBSCAN(eps=high_eps, min_samples=self.min_samples).fit(high_prices)
                
                # Group extrema by cluster
                for i, label in enumerate(high_clusters.labels_):
                    if label == -1:  # Noise points
                        continue
                    
                    level = high_prices[i][0]
                    if label not in clustered_highs:
                        clustered_highs[label] = {
                            'price': level,
                            'extrema': []
                        }
                    
                    clustered_highs[label]['extrema'].append(high_extrema[i])
                
                # Calculate representative price for each cluster
                for label, cluster in clustered_highs.items():
                    prices = [ext['price'] for ext in cluster['extrema']]
                    volumes = [ext['volume'] for ext in cluster['extrema']]
                    
                    # Use volume-weighted average price
                    if sum(volumes) > 0:
                        cluster['price'] = sum(p * v for p, v in zip(prices, volumes)) / sum(volumes)
                    else:
                        cluster['price'] = sum(prices) / len(prices)
        
        # Cluster low prices if we have enough points
        clustered_lows = {}
        if len(low_prices) >= self.min_samples:
            # Scale eps by price level for percentage-based clustering
            if len(low_prices) > 0:
                mean_price = np.mean(low_prices)
                low_eps = mean_price * self.cluster_eps
                
                # Cluster low extrema
                low_clusters = DBSCAN(eps=low_eps, min_samples=self.min_samples).fit(low_prices)
                
                # Group extrema by cluster
                for i, label in enumerate(low_clusters.labels_):
                    if label == -1:  # Noise points
                        continue
                    
                    level = low_prices[i][0]
                    if label not in clustered_lows:
                        clustered_lows[label] = {
                            'price': level,
                            'extrema': []
                        }
                    
                    clustered_lows[label]['extrema'].append(low_extrema[i])
                
                # Calculate representative price for each cluster
                for label, cluster in clustered_lows.items():
                    prices = [ext['price'] for ext in cluster['extrema']]
                    volumes = [ext['volume'] for ext in cluster['extrema']]
                    
                    # Use volume-weighted average price
                    if sum(volumes) > 0:
                        cluster['price'] = sum(p * v for p, v in zip(prices, volumes)) / sum(volumes)
                    else:
                        cluster['price'] = sum(prices) / len(prices)
        
        # Convert to price->score format
        high_levels = {cluster['price']: cluster['extrema'] for cluster in clustered_highs.values()}
        low_levels = {cluster['price']: cluster['extrema'] for cluster in clustered_lows.values()}
        
        return high_levels, low_levels
    
    def _calculate_level_scores(self, level_extrema, data):
        """
        Calculate strength scores for each level based on:
        1. Number of touches
        2. Timeframe weight
        3. Volume at touch points
        
        Args:
            level_extrema (dict): Dictionary of level prices to extrema
            data (pd.DataFrame): Original OHLCV data for context
            
        Returns:
            dict: Dictionary of level prices to scores
        """
        scores = {}
        
        # Process each level
        for level, extrema in level_extrema.items():
            # Base score starts with number of touches
            touch_count = len(extrema)
            
            # Calculate timeframe component
            timeframe_score = 0
            tf_counts = {}
            
            for ext in extrema:
                tf = ext['timeframe']
                tf_counts[tf] = tf_counts.get(tf, 0) + 1
            
            # Weight by timeframe importance
            for tf, count in tf_counts.items():
                weight = self.timeframe_weights.get(tf, 1.0)
                timeframe_score += count * weight
            
            # Calculate volume component
            volume_score = 0
            if self.volume_weight > 0:
                # Normalize volumes relative to average volume
                avg_volume = np.mean([ext['volume'] for ext in extrema])
                if avg_volume > 0:
                    for ext in extrema:
                        rel_volume = ext['volume'] / avg_volume
                        volume_score += rel_volume
                
                # Scale by volume weight
                volume_score *= self.volume_weight
            
            # Calculate recency factor
            # More recent touches are weighted higher
            recency_factor = 1.0
            if len(extrema) > 0:
                # Get timestamps
                timestamps = [ext['timestamp'] for ext in extrema]
                
                # Sort by recency
                sorted_timestamps = sorted(timestamps)
                
                # Most recent timestamp
                latest = sorted_timestamps[-1]
                
                # Calculate recency weights
                recency_weights = []
                for ts in timestamps:
                    # Convert to datetime if not already
                    if not isinstance(ts, pd.Timestamp):
                        ts = pd.Timestamp(ts)
                    if not isinstance(latest, pd.Timestamp):
                        latest = pd.Timestamp(latest)
                    
                    # Calculate days difference
                    days_diff = (latest - ts).days
                    
                    # Weight decays with age
                    weight = np.exp(-days_diff / 30)  # Decay factor of 30 days
                    recency_weights.append(weight)
                
                recency_factor = sum(recency_weights) / len(recency_weights)
            
            # Calculate final score
            final_score = (touch_count + timeframe_score + volume_score) * recency_factor
            
            # Store the score
            scores[level] = final_score
        
        return scores
    
    def plot(self, ax, data, major_support_color='darkgreen', minor_support_color='lightgreen',
             major_resistance_color='darkred', minor_resistance_color='lightcoral',
             linewidth=1.5, alpha=0.7, show_scores=True):
        """
        Plot support and resistance levels on the given matplotlib axis.
        
        Args:
            ax (matplotlib.axes.Axes): The axis to plot on
            data (pd.DataFrame): The OHLCV data
            major_support_color (str): Color for major support levels
            minor_support_color (str): Color for minor support levels
            major_resistance_color (str): Color for major resistance levels
            minor_resistance_color (str): Color for minor resistance levels
            linewidth (float): Line width
            alpha (float): Line transparency
            show_scores (bool): Whether to show level scores in the plot
            
        Returns:
            list: List of artists added to the plot
        """
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        
        if not self.is_calculated:
            raise ValueError("Indicator hasn't been calculated yet. Call calculate() first.")
            
        artists = []
        
        # Plot major resistance levels
        for level in self.values['major_resistance']:
            score = self.values['resistance_scores'][level]
            label = f"Major R: {level:.2f}" if level == self.values['major_resistance'][0] else ""
            if show_scores:
                label += f" ({score:.1f})" if label else f"({score:.1f})"
                
            line = ax.axhline(level, color=major_resistance_color, linestyle='-', 
                             linewidth=linewidth, alpha=alpha, label=label)
            artists.append(line)
            
        # Plot minor resistance levels
        for level in self.values['minor_resistance']:
            score = self.values['resistance_scores'][level]
            label = f"Minor R: {level:.2f}" if level == self.values['minor_resistance'][0] else ""
            if show_scores:
                label += f" ({score:.1f})" if label else f"({score:.1f})"
                
            line = ax.axhline(level, color=minor_resistance_color, linestyle='--', 
                             linewidth=linewidth-0.5, alpha=alpha-0.1, label=label)
            artists.append(line)
            
        # Plot major support levels
        for level in self.values['major_support']:
            score = self.values['support_scores'][level]
            label = f"Major S: {level:.2f}" if level == self.values['major_support'][0] else ""
            if show_scores:
                label += f" ({score:.1f})" if label else f"({score:.1f})"
                
            line = ax.axhline(level, color=major_support_color, linestyle='-', 
                             linewidth=linewidth, alpha=alpha, label=label)
            artists.append(line)
            
        # Plot minor support levels
        for level in self.values['minor_support']:
            score = self.values['support_scores'][level]
            label = f"Minor S: {level:.2f}" if level == self.values['minor_support'][0] else ""
            if show_scores:
                label += f" ({score:.1f})" if label else f"({score:.1f})"
                
            line = ax.axhline(level, color=minor_support_color, linestyle='--', 
                             linewidth=linewidth-0.5, alpha=alpha-0.1, label=label)
            artists.append(line)
            
        return artists 