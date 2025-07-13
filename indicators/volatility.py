import numpy as np
import pandas as pd
from .base_indicator import BaseIndicator

class BollingerBands(BaseIndicator):
    """
    Bollinger Bands indicator.
    
    This indicator calculates a moving average and upper/lower bands based on
    standard deviation, useful for identifying overbought/oversold conditions
    and volatility.
    """
    
    def __init__(self, window=20, num_std=2.0, column='Close'):
        """
        Initialize the Bollinger Bands indicator.
        
        Args:
            window (int): Window size for the moving average
            num_std (float): Number of standard deviations for the bands
            column (str): Column to use for calculations
            name (str): Name of this indicator instance
        """
        self.window = window
        self.num_std = num_std
        self.column = column

        super().__init__()
    
    def get_column_names(self):
        """Return column names produced by this indicator."""
        return [
            f'BollingerBands_{self.column}_{self.window}_std_{self.num_std}_Middle',
            f'BollingerBands_{self.column}_{self.window}_std_{self.num_std}_Upper',
            f'BollingerBands_{self.column}_{self.window}_std_{self.num_std}_Lower'
        ]
    
    def _calculate_for_single_df(self, data, append=True, **kwargs):
        """
        Calculate Bollinger Bands.
        
        Args:
            data (pd.DataFrame): OHLCV data
            append (bool): Whether to append results to original data
            **kwargs: Additional parameters (ignored)
            
        Returns:
            pd.DataFrame: DataFrame with Bollinger Bands values
        """
        # Check if the required column exists
        if self.column not in data.columns:
            raise ValueError(f"Column '{self.column}' not found in data")
        
        # Calculate the middle band (simple moving average)
        middle_band = data[self.column].rolling(window=self.window).mean()
        
        # Calculate the standard deviation
        std_dev = data[self.column].rolling(window=self.window).std()
        
        # Calculate upper and lower bands
        upper_band = middle_band + (std_dev * self.num_std)
        lower_band = middle_band - (std_dev * self.num_std)
        
        # Store the calculated values
        indicator_data = {
            f'{self.column_names[0]}': middle_band,
            f'{self.column_names[1]}': upper_band,
            f'{self.column_names[2]}': lower_band,
            #f'{self.column_names[3]}': (upper_band - lower_band) / middle_band  # Normalized width
        }
        
        self.is_calculated = True
        
        # Return results according to the append parameter
        return self._append_to_df(data, indicator_data) if append else self._create_indicator_df(data, indicator_data)
    
'''

class ATR(BaseIndicator):
    """
    Average True Range (ATR) indicator.
    
    This indicator measures market volatility by decomposing the entire range 
    of an asset for a given period.
    """
    
    def __init__(self, window=14):
        """
        Initialize the ATR indicator.
        
        Args:
            window (int): Window size for the average calculation
            name (str): Name of this indicator instance
        """
        super().__init__()
        self.window = window
    
    def get_column_names(self):
        """Return column names produced by this indicator."""
        return [
            f'ATR_{self.window}'
        ]
    
    def _calculate_for_single_df(self, data, append=True, **kwargs):
        """
        Calculate the Average True Range.
        
        Args:
            data (pd.DataFrame): OHLCV data
            append (bool): Whether to append results to original data
            **kwargs: Additional parameters (ignored)
            
        Returns:
            pd.DataFrame: DataFrame with ATR values
        """
        # Check if required columns exist
        required_columns = ['High', 'Low', 'Close']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        # Calculate true range
        high = data['High']
        low = data['Low']
        close = data['Close'].shift(1)  # Previous close
        
        # Handle first row where previous close is not available
        close.iloc[0] = data['Open'].iloc[0] if 'Open' in data.columns else (high.iloc[0] + low.iloc[0]) / 2
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        true_range = pd.DataFrame({
            'tr1': tr1,
            'tr2': tr2,
            'tr3': tr3
        }).max(axis=1)
        
        # Calculate ATR
        atr = true_range.rolling(window=self.window).mean()
        
        # Store the calculated values
        self.values = {
            f'{self.name}': atr,
            f'{self.name}_TrueRange': true_range
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
    
    

class VolatilityRatio(BaseIndicator):
    """
    Volatility Ratio indicator.
    
    This indicator compares recent volatility to historical volatility
    to identify volatility expansion and contraction regimes.
    """
    
    def __init__(self, fast_window=5, slow_window=20, atr_window=14, name="Volatility_Ratio"):
        """
        Initialize the Volatility Ratio indicator.
        
        Args:
            fast_window (int): Window size for recent volatility measurement
            slow_window (int): Window size for historical volatility measurement
            atr_window (int): Window size for ATR calculation
            name (str): Name of this indicator instance
        """
        super().__init__(name=name)
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.atr_window = atr_window
    
    def calculate(self, data, append=True, **kwargs):
        """
        Calculate the Volatility Ratio.
        
        Args:
            data (pd.DataFrame): OHLCV data
            append (bool): Whether to append results to original data
            **kwargs: Additional parameters (ignored)
            
        Returns:
            pd.DataFrame: DataFrame with Volatility Ratio values
        """
        # Calculate ATR for both windows
        atr = ATR(window=self.atr_window)
        atr_data = atr.calculate(data, append=False)
        atr_values = atr_data[f'ATR']
        
        # Calculate fast and slow volatility
        fast_vol = atr_values.rolling(window=self.fast_window).mean()
        slow_vol = atr_values.rolling(window=self.slow_window).mean()
        
        # Calculate ratio
        vol_ratio = fast_vol / slow_vol
        
        # Store the calculated values
        self.values = {
            f'{self.name}': vol_ratio,
            f'{self.name}_Fast': fast_vol,
            f'{self.name}_Slow': slow_vol
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
    
'''