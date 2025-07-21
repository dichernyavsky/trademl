"""
Volume Indicators from TA-Lib

This module contains volume-based indicators that can be normalized
for machine learning applications.
"""

import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Optional, Tuple, Union

from .base_talib_indicator import BaseTALibIndicator


class OBV(BaseTALibIndicator):
    """
    On Balance Volume (OBV)
    
    OBV measures buying and selling pressure as a cumulative indicator
    that adds volume on up days and subtracts it on down days.
    
    Parameters:
        normalize (bool): Whether to normalize values to 0-1 range (default: True)
    """
    
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
    
    def get_column_names(self) -> List[str]:
        return ['OBV']
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate OBV
        
        Args:
            data: DataFrame with 'close', 'volume' columns
            
        Returns:
            DataFrame with OBV column added
        """
        required_cols = ['close', 'volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        result = data.copy()
        obv_values = talib.OBV(data['close'].values, data['volume'].values)
        
        if self.normalize:
            # Normalize OBV using min-max scaling
            obv_min = np.nanmin(obv_values)
            obv_max = np.nanmax(obv_values)
            if obv_max > obv_min:
                obv_values = (obv_values - obv_min) / (obv_max - obv_min)
            else:
                obv_values = np.zeros_like(obv_values)
        
        result['OBV'] = obv_values
        
        return result


class AccumulationDistribution(BaseTALibIndicator):
    """
    Accumulation/Distribution Line
    
    The A/D line measures the cumulative flow of money into and out of a security.
    It combines price and volume to show how money may be flowing into or out of a stock.
    
    Parameters:
        normalize (bool): Whether to normalize values to 0-1 range (default: True)
    """
    
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
    
    def get_column_names(self) -> List[str]:
        return ['AD']
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Accumulation/Distribution
        
        Args:
            data: DataFrame with 'high', 'low', 'close', 'volume' columns
            
        Returns:
            DataFrame with AD column added
        """
        required_cols = ['high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        result = data.copy()
        ad_values = talib.AD(data['high'].values, data['low'].values,
                            data['close'].values, data['volume'].values)
        
        if self.normalize:
            # Normalize AD using min-max scaling
            ad_min = np.nanmin(ad_values)
            ad_max = np.nanmax(ad_values)
            if ad_max > ad_min:
                ad_values = (ad_values - ad_min) / (ad_max - ad_min)
            else:
                ad_values = np.zeros_like(ad_values)
        
        result['AD'] = ad_values
        
        return result


class ChaikinMoneyFlow(BaseTALibIndicator):
    """
    Chaikin Money Flow (CMF)
    
    CMF measures the amount of money flow volume over a specific period.
    It ranges from -1 to +1, where positive values indicate buying pressure.
    
    Parameters:
        period (int): Number of periods for calculation (default: 20)
        normalize (bool): Whether to normalize values to 0-1 range (default: True)
    """
    
    def __init__(self, period: int = 20, normalize: bool = True):
        super().__init__()
        self.period = period
        self.normalize = normalize
    
    def get_column_names(self) -> List[str]:
        return [f'CMF_{self.period}']
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Chaikin Money Flow
        
        Args:
            data: DataFrame with 'high', 'low', 'close', 'volume' columns
            
        Returns:
            DataFrame with CMF column added
        """
        required_cols = ['high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        result = data.copy()
        
        # Calculate Money Flow Multiplier
        mfm = ((data['close'].values - data['low'].values) - (data['high'].values - data['close'].values)) / (data['high'].values - data['low'].values)
        mfm = np.where(data['high'].values == data['low'].values, 0, mfm)
        
        # Calculate Money Flow Volume
        mfv = mfm * data['volume'].values
        
        # Calculate CMF as rolling sum of MFV / rolling sum of volume
        cmf_values = pd.Series(mfv).rolling(window=self.period).sum() / pd.Series(data['volume'].values).rolling(window=self.period).sum()
        cmf_values = cmf_values.values
        
        if self.normalize:
            # CMF ranges from -1 to +1, normalize to 0-1
            cmf_values = (cmf_values + 1) / 2
            cmf_values = np.clip(cmf_values, 0, 1)
        
        result[f'CMF_{self.period}'] = cmf_values
        
        return result


class MFI(BaseTALibIndicator):
    """
    Money Flow Index (MFI)
    
    MFI is a momentum indicator that measures the money flow (volume and price)
    into and out of a security. It ranges from 0 to 100.
    
    Parameters:
        period (int): Number of periods for calculation (default: 14)
        normalize (bool): Whether to normalize values to 0-1 range (default: True)
    """
    
    def __init__(self, period: int = 14, normalize: bool = True):
        super().__init__()
        self.period = period
        self.normalize = normalize
    
    def get_column_names(self) -> List[str]:
        return [f'MFI_{self.period}']
    
    def _calculate_talib(self, data: pd.DataFrame, **kwargs) -> Dict[str, np.ndarray]:
        """
        Calculate MFI using TA-Lib.
        
        Args:
            data: DataFrame with TA-Lib standard column names
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with MFI values
        """
        self._validate_required_columns(data, ['high', 'low', 'close', 'volume'])
        
        mfi_values = talib.MFI(data['high'].values, data['low'].values,
                              data['close'].values, data['volume'].values, timeperiod=self.period)
        
        if self.normalize:
            # MFI is already 0-100, normalize to 0-1
            mfi_values = mfi_values / 100.0
        
        return {f'MFI_{self.period}': mfi_values} 