import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from .base_indicator import BaseIndicator

class ClusterSupportResistance(BaseIndicator):
    """
    Support and resistance levels indicator based on price clustering.
    
    This indicator uses machine learning clustering algorithms to identify
    significant price levels where the price has spent considerable time.
    """
    
    def __init__(self, n_levels=10, method='kmeans', lookback=None, eps=0.005, 
                min_samples=5, max_levels=5, value_area_volume=70):
        """
        Initialize the support/resistance indicator.
        
        Args:
            n_levels (int): Number of price levels to detect (for KMeans)
            method (str): Clustering method ('kmeans' or 'dbscan')
            lookback (int, optional): Number of bars to look back, None for all data
            eps (float): The maximum distance between samples for DBSCAN
            min_samples (int): Minimum number of samples in a cluster for DBSCAN
            max_levels (int): Maximum number of support/resistance levels to return
            value_area_volume (float): Percentage of volume to include in value area (0-100)
        """
        super().__init__(name="Cluster Support Resistance")
        self.n_levels = n_levels
        self.method = method
        self.lookback = lookback
        self.eps = eps
        self.min_samples = min_samples
        self.max_levels = max_levels
        self.value_area_volume = value_area_volume / 100  # Convert to proportion
    
    def calculate(self, data):
        """
        Calculate support and resistance levels using clustering.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            dict: Dictionary with 'support_levels' and 'resistance_levels'
        """
        # Extract price and volume data
        if self.lookback is not None and self.lookback < len(data):
            data = data.iloc[-self.lookback:]
            
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        volume = data.get('Volume', pd.Series(np.ones_like(close))).values
        
        # Create price points for clustering
        # We'll use highs, lows and closes with their respective volumes
        price_points = []
        volumes = []
        
        for i in range(len(close)):
            # Add close price (highest weight)
            price_points.append(close[i])
            volumes.append(volume[i] * 2)  # Give double weight to closes
            
            # Add high and low
            price_points.append(high[i])
            price_points.append(low[i])
            volumes.append(volume[i])
            volumes.append(volume[i])
        
        price_points = np.array(price_points).reshape(-1, 1)
        volumes = np.array(volumes)
        
        # Apply clustering based on selected method
        if self.method == 'kmeans':
            levels = self._kmeans_clustering(price_points)
        else:  # 'dbscan'
            levels = self._dbscan_clustering(price_points)
        
        # Sort levels and add their strength
        strength = self._calculate_level_strength(levels, price_points, volumes)
        
        # Sort levels by their strength
        level_strength = sorted(zip(levels, strength), key=lambda x: x[1], reverse=True)
        
        # Select only the strongest levels, up to max_levels
        top_levels = level_strength[:min(self.max_levels * 2, len(level_strength))]
        levels = [level for level, _ in top_levels]
        
        # Determine which levels are support and which are resistance
        current_price = close[-1]
        resistance_levels = sorted([l for l in levels if l > current_price])
        support_levels = sorted([l for l in levels if l <= current_price], reverse=True)
        
        # If we have too many levels, trim them
        resistance_levels = resistance_levels[:self.max_levels]
        support_levels = support_levels[:self.max_levels]
        
        # Create arrays for plotting in strategy
        support_lines = np.full(len(close), np.nan)
        resistance_lines = np.full(len(close), np.nan)
        
        # Create time series for each level
        result = {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'all_levels': levels,
            'level_strength': dict(level_strength)
        }
        
        self.values = result
        self.is_calculated = True
        
        return result
    
    def _kmeans_clustering(self, price_points):
        """Apply KMeans clustering to find levels"""
        kmeans = KMeans(n_clusters=self.n_levels, random_state=42)
        kmeans.fit(price_points)
        levels = kmeans.cluster_centers_.flatten()
        return sorted(levels)
    
    def _dbscan_clustering(self, price_points):
        """Apply DBSCAN clustering to find levels"""
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        clusters = dbscan.fit_predict(price_points)
        
        # For each valid cluster, calculate its center
        levels = []
        for cluster_id in set(clusters):
            if cluster_id != -1:  # -1 is noise in DBSCAN
                cluster_points = price_points[clusters == cluster_id]
                level = np.mean(cluster_points)
                levels.append(level)
        
        return sorted(levels)
    
    def _calculate_level_strength(self, levels, price_points, volumes):
        """
        Calculate the strength of each level based on:
        1. How much time price spent near the level
        2. How much volume was traded near the level
        
        Args:
            levels (list): List of price levels
            price_points (np.ndarray): Array of price points
            volumes (np.ndarray): Array of volumes corresponding to price points
            
        Returns:
            list: Strength score for each level
        """
        strength = []
        
        # Define a window around each level
        for level in levels:
            # Consider points within 0.5% of the level
            window = 0.005 * level
            
            # Find points that are within the window
            mask = (price_points >= level - window) & (price_points <= level + window)
            mask = mask.flatten()
            
            # If we have volume data, use it for weighting
            if volumes is not None:
                # Calculate weighted count (points weighted by volume)
                level_strength = np.sum(volumes[mask])
            else:
                # Simple count of points near the level
                level_strength = np.sum(mask)
            
            strength.append(level_strength)
            
        return strength

    


class SimpleSupportResistance(BaseIndicator):
    """
    Simple Support and Resistance indicator that identifies pivot points
    and extends them as support/resistance levels until broken.
    """
    
    def __init__(self, lookback=20, high_col='High', low_col='Low', name="SimpleSR"):
        """
        Initialize the Support/Resistance indicator.
        
        Args:
            lookback (int): Number of bars to look before and after for pivot points
            high_col (str): Column name to use for pivot highs calculation
            low_col (str): Column name to use for pivot lows calculation
            name (str): Indicator name
        """
        super().__init__(name=name)
        self.lookback = lookback
        self.high_col = high_col
        self.low_col = low_col

    def get_column_names(self, **kwargs):
        """Return column names produced by this indicator."""
        lookback = kwargs.get('lookback', self.lookback)
        return [
            f'SimpleSR_{lookback}_Pivot_High',
            f'SimpleSR_{lookback}_Pivot_Low', 
            f'SimpleSR_{lookback}_Resistance',
            f'SimpleSR_{lookback}_Support'
        ]
    
    def calculate(self, data, append=True, **kwargs):
        """
        Calculate pivot points and support/resistance levels.
        
        Args:
            data (pd.DataFrame): OHLCV DataFrame
            append (bool): Whether to append results to original DataFrame or return new one
            **kwargs: Additional parameters (can override instance variables)
            
        Returns:
            pd.DataFrame: DataFrame with pivot points and support/resistance levels
        """
        # Apply kwargs if provided
        lookback = kwargs.get('lookback', self.lookback)
        high_col = kwargs.get('high_col', self.high_col)
        low_col = kwargs.get('low_col', self.low_col)
        
        # Make a working copy of the data
        df = data.copy()
        
        # Find pivot points
        df['PH'] = self._find_pivot_highs(df, lookback, high_col)
        df['PL'] = self._find_pivot_lows(df, lookback, low_col)
        
        # Extend the pivot lines
        df['Resistance'] = self._extend_pivot_high_line(df)
        df['Support'] = self._extend_pivot_low_line(df)
        
        # Store calculated values
        self.values = {
            f'SimpleSR_{lookback}_Pivot_High': df['PH'],
            f'SimpleSR_{lookback}_Pivot_Low': df['PL'],
            f'SimpleSR_{lookback}_Resistance': df['Resistance'],
            f'SimpleSR_{lookback}_Support': df['Support']
        }
        self.is_calculated = True
        
        # Return results according to the append parameter
        if append:
            result = data.copy()
            for key, values in self.values.items():
                result[key] = values
            return result
        else:
            return pd.DataFrame(self.values, index=data.index)
    
    def _find_pivot_highs(self, df, lookback, high_col='High'):
        """
        Identify pivot high points in the specified column of the DataFrame.
        
        A pivot high is defined as a bar where the value is greater than the maximum
        of the previous `lookback` bars and the maximum of the following `lookback` bars.
        
        Args:
            df (DataFrame): Input DataFrame
            lookback (int): The number of bars to consider before and after
            high_col (str): The column to use for pivot high calculation
            
        Returns:
            Series: A Series containing pivot high values or NaN
        """
        high_values = df[high_col]
        
        # Compute the maximum of the previous `lookback` bars (excluding the current bar)
        left_window_max = high_values.shift(1).rolling(window=lookback, min_periods=lookback).max()
        
        # Compute the maximum of the next `lookback` bars (excluding the current bar)
        right_window_max = high_values[::-1].shift(1).rolling(window=lookback, min_periods=lookback).max()[::-1]
        
        # Identify pivot highs where the current high is greater than both left and right window maximums
        pivot_highs = high_values.where((high_values > left_window_max) & (high_values > right_window_max))
        return pivot_highs
    
    def _find_pivot_lows(self, df, lookback, low_col='Low'):
        """
        Identify pivot low points in the specified column of the DataFrame.
        
        A pivot low is defined as a bar where the value is lower than the minimum
        of the previous `lookback` bars and the minimum of the following `lookback` bars.
        
        Args:
            df (DataFrame): Input DataFrame
            lookback (int): The number of bars to consider before and after
            low_col (str): The column to use for pivot low calculation
            
        Returns:
            Series: A Series containing pivot low values or NaN
        """
        low_values = df[low_col]
        
        # Compute the minimum of the previous `lookback` bars (excluding the current bar)
        left_window_min = low_values.shift(1).rolling(window=lookback, min_periods=lookback).min()
        
        # Compute the minimum of the next `lookback` bars (excluding the current bar)
        right_window_min = low_values[::-1].shift(1).rolling(window=lookback, min_periods=lookback).min()[::-1]
        
        # Identify pivot lows where the current low is lower than both left and right window minimums
        pivot_lows = low_values.where((low_values < left_window_min) & (low_values < right_window_min))
        return pivot_lows
    
    def _extend_pivot_high_line(self, df):
        """
        Extend the pivot high line as resistance until broken.
        
        Args:
            df (DataFrame): DataFrame with 'PH' column containing pivot high points
            
        Returns:
            Series: Extended resistance line
        """
        # Create groups where each group starts at a pivot high (non-null 'PH')
        pivot_groups = df['PH'].notna().cumsum()
        
        # For each group, get the first 'High' value as the baseline pivot (p1)
        baseline_high = df.groupby(pivot_groups)[self.high_col].transform('first')
        
        # Identify the first row in each group (this row is not subject to violation check)
        is_first_in_group = df.groupby(pivot_groups).cumcount() == 0
        
        # For subsequent rows, mark as violation if the current high is greater than or equal to the baseline
        violation = (~is_first_in_group) & (df[self.high_col] >= baseline_high)
        
        # Propagate the violation flag within each group: once a violation occurs, flag all following bars
        violation_cum = violation.groupby(pivot_groups).cummax()
        
        # Set the pivot high line to the baseline until a violation occurs; thereafter, assign NaN
        extended_line = baseline_high.where(~violation_cum, np.nan)
        
        return extended_line
    
    def _extend_pivot_low_line(self, df):
        """
        Extend the pivot low line as support until broken.
        
        Args:
            df (DataFrame): DataFrame with 'PL' column containing pivot low points
            
        Returns:
            Series: Extended support line
        """
        # Create groups where each group starts at a pivot low (non-null 'PL')
        pivot_groups = df['PL'].notna().cumsum()
        
        # For each group, get the first 'Low' value as the baseline pivot low
        baseline_low = df.groupby(pivot_groups)[self.low_col].transform('first')
        
        # Identify the first row in each group (this row is not subject to violation check)
        is_first_in_group = df.groupby(pivot_groups).cumcount() == 0
        
        # For subsequent rows, mark as violation if the current low is less than or equal to the baseline
        violation = (~is_first_in_group) & (df[self.low_col] <= baseline_low)
        
        # Propagate the violation flag within each group: once a violation occurs, flag all following bars
        cumulative_violation = violation.groupby(pivot_groups).cummax()
        
        # Set the pivot low line to the baseline until a violation occurs; thereafter, assign NaN
        extended_line = baseline_low.where(~cumulative_violation, np.nan)
        
        return extended_line 