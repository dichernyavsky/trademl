import pandas as pd
from .base_indicator import BaseIndicator

class VolumeMA(BaseIndicator):
    """
    Volume Moving Average indicator.
    """

    def __init__(self, window=20, name="Volume_MA"):
        """
        Initialize the Volume MA indicator.
        
        Args:
            window (int): Window size for the moving average
            name (str): Name of this indicator instance
        """
        super().__init__(name=name)
        self.window = window
    
    def calculate(self, data, append=True, **kwargs):   
        # Check if Volume column exists
        if 'Volume' not in data.columns:
            raise ValueError("Volume column not found in data")
        
        # Calculate the moving average
        volume_ma = data['Volume'].rolling(window=self.window).mean()
        
        # Store the calculated values
        self.values = {
            f'{self.name}': volume_ma
        }
        
        self.is_calculated = True
        
        # Return results according to the append parameter
        if append:
            result = data.copy()
            result[self.name] = volume_ma
            return result
        else:
            return pd.DataFrame(self.values, index=data.index)
    