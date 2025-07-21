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
        
        # Get column names
        column_names = self.get_column_names()
        
        # Store the calculated values
        indicator_data = {
            column_names[0]: middle_band,
            column_names[1]: upper_band,
            column_names[2]: lower_band,
        }
        
        self.values = indicator_data
        self.is_calculated = True
        
        # Return results according to the append parameter
        return self._append_to_df(data, indicator_data) if append else self._create_indicator_df(data, indicator_data)
    


class ATRIndicator(BaseIndicator):
    """Average True Range (ATR) indicator.

    Parameters
    ----------
    window : int, default 14
        Lookback period for ATR calculation.
    method : {"wilder", "sma"}, default "wilder"
        - "wilder": exponential moving average with  alpha = 1 / window (classic ATR).
        - "sma": simple moving average of True Range.
    high_col, low_col, close_col : str
        Column names in the OHLCV DataFrame.
    name : str or None
        Column prefix; if None, generated automatically.
    """

    def __init__(self,
                 window: int = 14,
                 method: str = "wilder",
                 high_col: str = "High",
                 low_col: str = "Low",
                 close_col: str = "Close",
                 name: str = None):
                 
        self.window = window
        self.method = method.lower()
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self._name = name or f"ATR_{window}"
        super().__init__()

    def get_column_names(self, **kwargs):
        return [self._name]

    def _calculate_for_single_df(self, data: pd.DataFrame, append: bool = True, **kwargs):
        if data.empty:
            return data

        # Validate required columns
        for col in (self.high_col, self.low_col, self.close_col):
            if col not in data.columns:
                raise ValueError(f"Column '{col}' missing from input DataFrame")

        high = data[self.high_col]
        low = data[self.low_col]
        prev_close = data[self.close_col].shift(1)

        # On the first bar replace NaN prev_close with mean of high/low
        if prev_close.isna().iat[0]:
            prev_close = prev_close.copy()
            prev_close.iat[0] = (high.iat[0] + low.iat[0]) / 2

        # True Range components
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR smoothing
        if self.method == "sma":
            atr = true_range.rolling(window=self.window, min_periods=self.window).mean()
        elif self.method == "wilder":
            alpha = 1 / self.window
            atr = true_range.ewm(alpha=alpha, adjust=False).mean()
        else:
            raise ValueError("method must be 'wilder' or 'sma'")

        indicator_data = {self._name: atr}
        self.values = indicator_data
        self.is_calculated = True

        return (self._append_to_df(data, indicator_data)
                if append else self._create_indicator_df(data, indicator_data))
    
    

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
    
