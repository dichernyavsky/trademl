import numpy as np
import pandas as pd
from .base_indicator import BaseIndicator

class VolumeProfileSupportResistance(BaseIndicator):
    """
    Support and resistance indicator based on Volume Profile.
    
    This indicator creates a volume profile (price histogram weighted by volume)
    and identifies key levels where significant volume has been traded.
    """
    
    def __init__(self, n_bins=100, poc_threshold=80, va_threshold=70, 
                 lookback=None, max_levels=3):
        """
        Initialize the Volume Profile indicator.
        
        Args:
            n_bins (int): Number of price bins for the volume profile
            poc_threshold (float): Percentile threshold for POC (0-100)
            va_threshold (float): Percentile threshold for Value Area (0-100)
            lookback (int, optional): Number of bars to look back, None for all data
            max_levels (int): Maximum number of levels to return
        """
        super().__init__(name="Volume Profile")
        self.n_bins = n_bins
        self.poc_threshold = poc_threshold / 100  # Convert to proportion
        self.va_threshold = va_threshold / 100  # Convert to proportion
        self.lookback = lookback
        self.max_levels = max_levels
    
    def calculate(self, data):
        """
        Calculate the volume profile and identify support/resistance levels.
        
        Args:
            data (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            dict: Dictionary with key levels and volume profile data
        """
        # Extract data for the specified lookback period
        if self.lookback is not None and self.lookback < len(data):
            data = data.iloc[-self.lookback:]
            
        # Extract price and volume data
        high = data['High'].values
        low = data['Low'].values
        close = data['Close'].values
        volume = data.get('Volume', pd.Series(np.ones_like(close))).values
        
        # Determine price range for the volume profile
        min_price = np.min(low)
        max_price = np.max(high)
        price_range = max_price - min_price
        
        # Create price bins
        bin_edges = np.linspace(min_price, max_price, self.n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Create volume profile
        volume_profile = np.zeros(self.n_bins)
        
        # Distribute volume across price range of each bar
        for i in range(len(data)):
            bar_low = low[i]
            bar_high = high[i]
            bar_volume = volume[i]
            
            # Find price bins that overlap with this bar
            bin_indices = np.where((bin_centers >= bar_low) & (bin_centers <= bar_high))[0]
            
            if len(bin_indices) > 0:
                # Distribute volume equally across overlapping bins
                volume_per_bin = bar_volume / len(bin_indices)
                volume_profile[bin_indices] += volume_per_bin
        
        # Find Point of Control (POC) - price with highest volume
        poc_index = np.argmax(volume_profile)
        poc_price = bin_centers[poc_index]
        
        # Find Value Area - price range containing specified percentage of volume
        sorted_indices = np.argsort(volume_profile)[::-1]  # Sort by volume, descending
        cumulative_volume = np.cumsum(volume_profile[sorted_indices])
        total_volume = cumulative_volume[-1]
        
        # Find bins in the value area
        va_indices = sorted_indices[cumulative_volume <= total_volume * self.va_threshold]
        
        # Get high-volume nodes based on poc_threshold
        volume_threshold = np.percentile(volume_profile, self.poc_threshold)
        high_volume_indices = np.where(volume_profile >= volume_threshold)[0]
        high_volume_prices = bin_centers[high_volume_indices]
        
        # Sort high volume prices by volume
        high_volume_strengths = volume_profile[high_volume_indices]
        high_volume_sorted = sorted(zip(high_volume_prices, high_volume_strengths), 
                                    key=lambda x: x[1], reverse=True)
        
        high_volume_levels = [price for price, _ in high_volume_sorted]
        
        # Separate into support and resistance levels
        current_price = close[-1]
        
        resistance_levels = sorted([p for p in high_volume_levels if p > current_price])
        support_levels = sorted([p for p in high_volume_levels if p <= current_price], 
                               reverse=True)
        
        # Limit the number of levels to return
        resistance_levels = resistance_levels[:self.max_levels]
        support_levels = support_levels[:self.max_levels]
        
        # Calculate value area high and low
        if len(va_indices) > 0:
            va_high = np.max(bin_centers[va_indices])
            va_low = np.min(bin_centers[va_indices])
        else:
            va_high = poc_price * 1.005  # Default if no VA can be calculated
            va_low = poc_price * 0.995
        
        result = {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'poc': poc_price,
            'va_high': va_high,
            'va_low': va_low,
            'bin_centers': bin_centers,
            'volume_profile': volume_profile
        }
        
        self.values = result
        self.is_calculated = True
        
        return result
    