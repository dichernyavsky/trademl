"""
Various filtering techniques for financial time series data.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, List, Dict
from pykalman import KalmanFilter as PyKalmanFilter
from statsmodels.tsa.filters.hp_filter import hpfilter
from scipy import signal
import pywt
from PyEMD import EMD as PyEMD

class KalmanFilter:
    """
    Estimates hidden states in noisy time series data.
    """
    
    def __init__(self, 
                 transition_covariance: float = 0.01, 
                 observation_covariance: float = 0.1, 
                 initial_state_covariance: float = 1.0):
        """
        Initialize Kalman filter parameters.
        
        Parameters:
            transition_covariance: Process noise covariance
            observation_covariance: Measurement noise covariance
            initial_state_covariance: Initial state uncertainty
        """
        self.transition_covariance = transition_covariance
        self.observation_covariance = observation_covariance
        self.initial_state_covariance = initial_state_covariance
        
    def filter(self, data: Union[pd.Series, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parameters:
            data: Input time series data
            
        Returns:
            Tuple of (filtered_state_means, filtered_state_covariances)
        """
        # Convert to numpy array if pandas Series
        if isinstance(data, pd.Series):
            original_index = data.index
            data = data.values
        
        # Initialize filter
        kf = PyKalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=data[0],
            initial_state_covariance=self.initial_state_covariance,
            transition_covariance=self.transition_covariance,
            observation_covariance=self.observation_covariance
        )
        
        # Run filter
        state_means, state_covs = kf.filter(data)
        
        return state_means.flatten(), state_covs.flatten()
    
    def filter_series(self, series: pd.Series) -> pd.Series:
        """
        Params:
            series: Input time series
            
        Returns:
            Filtered time series
        """
        state_means, _ = self.filter(series)
        return pd.Series(state_means, index=series.index)


class HodrickPrescottFilter:
    """
    Hodrick-Prescott filter for trend-cycle decomposition.
    Separates a time series into trend and cyclical components.
    """
    
    def __init__(self, lambda_param: float = 1600):
        """
        Initialize HP filter.
        
        Parameters:
            lambda_param: Smoothing parameter 
                (1600 is quarterly data default, 14400 for monthly, 100 for annual)
        """
        self.lambda_param = lambda_param
    
    def filter(self, data: Union[pd.Series, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply HP filter to decompose series into trend and cycle.
        
        Parameters:
            data: Input time series
            
        Returns:
            Tuple of (trend_component, cyclical_component)
        """
        # Apply filter
        cycle, trend = hpfilter(data, lamb=self.lambda_param)
        
        return trend, cycle
    
    def filter_series(self, series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Apply HP filter and return results as pandas Series.
        
        Parameters:
            series: Input time series
            
        Returns:
            Tuple of (trend_series, cycle_series)
        """
        trend, cycle = self.filter(series)
        return pd.Series(trend, index=series.index), pd.Series(cycle, index=series.index)

class ButterworthFilter:
    """
    Butterworth filter 
    """
    
    def __init__(self, cutoff: float, fs: float = 1.0, order: int = 5, btype: str = 'low'):
        """
        Parameters:
            cutoff: Cutoff frequency (or frequencies for bandpass/bandstop)
            fs: Sampling frequency (defaults to 1.0 for normalized frequency)
            order: Filter order
            btype: Filter type ('low', 'high', 'band', or 'stop')
        """
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.btype = btype
        
        # Validate params
        if btype not in ['low', 'high', 'band', 'stop']:
            raise ValueError("btype must be one of: 'low', 'high', 'band', 'stop'")
    
    def filter(self, data: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """
        Apply Butterworth filter to time series.
        
        Parameters:
            data: Input time series
            
        Returns:
            Filtered data
        """
        # Convert to numpy array if pandas Series
        if isinstance(data, pd.Series):
            original_index = data.index
            data = data.values
        
        # Design filter
        nyquist = 0.5 * self.fs
        if self.btype in ['band', 'stop']:
            if not isinstance(self.cutoff, (list, tuple)) or len(self.cutoff) != 2:
                raise ValueError("For band and stop filters, cutoff must be a list or tuple of two frequencies")
            normal_cutoff = [c / nyquist for c in self.cutoff]
        else:
            normal_cutoff = self.cutoff / nyquist
            
        b, a = signal.butter(self.order, normal_cutoff, btype=self.btype, analog=False)
        
        # Apply filter
        filtered_data = signal.filtfilt(b, a, data)
        
        return filtered_data
    
    def filter_series(self, series: pd.Series) -> pd.Series:
        """
        Apply Butterworth filter and return as pandas Series.
        
        Parameters:
            series: Input time series
            
        Returns:
            Filtered time series
        """
        filtered_data = self.filter(series)
        return pd.Series(filtered_data, index=series.index)


class SavitzkyGolayFilter:
    """
    Savitzky-Golay filter for smoothing 
    """
    
    def __init__(self, window_length: int = 11, polyorder: int = 2):
        """
        Initialize Savitzky-Golay filter.
        
        Parameters:
            window_length: Length of the filter window (must be odd)
            polyorder: Order of the polynomial used for fitting
        """
        # Validate parameters
        if window_length % 2 == 0:
            raise ValueError("window_length must be odd")
        if polyorder >= window_length:
            raise ValueError("polyorder must be less than window_length")
            
        self.window_length = window_length
        self.polyorder = polyorder
    
    def filter_series(self, series: pd.Series) -> pd.Series:
        """
        Apply Savitzky-Golay filter to time series.
        
        Parameters:
            series: Input time series
            
        Returns:
            Filtered time series
        """
            
        filtered_data = signal.savgol_filter(
            series.values, 
            window_length=self.window_length, 
            polyorder=self.polyorder
        )
        
        return pd.Series(filtered_data, index=series.index)


class WaveletFilter:
    """
    Wavelet-based filtering for multi-resolution analysis of time series.
    
    Wavelet transforms decompose time series into different frequency components
    while preserving both frequency and time information, making them powerful
    for analyzing non-stationary financial data.
    """
    
    def __init__(self, wavelet: str = 'db8', level: int = 3, mode: str = 'symmetric'):
        """
        Initialize wavelet filter.
        
        Parameters:
            wavelet: Wavelet type ('db8', 'sym4', 'haar', etc.)
            level: Decomposition level
            mode: Signal extension mode
        """
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
    
    def decompose(self, series: pd.Series) -> Dict[str, pd.Series]:
        """
        Decompose time series into approximation and detail coefficients.
        
        Parameters:
            series: Input time series
            
        Returns:
            Dictionary with approximation and detail coefficient series
        """
        # Convert to numpy array
        data = series.values
        
        # Perform multilevel discrete wavelet transform
        coeffs = pywt.wavedec(data, self.wavelet, mode=self.mode, level=self.level)
        
        # Create dictionary with components
        result = {
            'approximation': pd.Series(coeffs[0], index=series.index[:len(coeffs[0])]),
        }
        
        for i in range(1, len(coeffs)):
            result[f'detail_{i}'] = pd.Series(coeffs[i], index=series.index[:len(coeffs[i])])
        
        return result
    
    def denoise(self, series: pd.Series, method: str = 'soft', threshold_scale: float = 1.0) -> pd.Series:
        """
        Denoise time series using wavelet thresholding.
        
        Parameters:
            series: Input time series
            method: Thresholding method ('soft' or 'hard')
            threshold_scale: Scaling factor for threshold
            
        Returns:
            Denoised time series
        """
        # Convert to numpy array
        data = series.values
        
        # Decompose
        coeffs = pywt.wavedec(data, self.wavelet, mode=self.mode, level=self.level)
        
        # Calculate threshold
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data))) * threshold_scale
        
        # Apply thresholding to detail coefficients only (keep approximation)
        new_coeffs = [coeffs[0]]
        for i in range(1, len(coeffs)):
            if method == 'soft':
                new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
            else:
                new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='hard'))
        
        # Reconstruct signal
        denoised_data = pywt.waverec(new_coeffs, self.wavelet, mode=self.mode)
        
        # Trim to original length if needed
        if len(denoised_data) > len(series):
            denoised_data = denoised_data[:len(series)]
        
        return pd.Series(denoised_data, index=series.index)
    
    def filter_series(self, series: pd.Series, keep_levels: List[int] = None) -> pd.Series:
        """
        Filter time series by reconstructing from selected decomposition levels.
        
        Parameters:
            series: Input time series
            keep_levels: Decomposition levels to keep (0=approximation, 1...n=detail levels)
                         If None, keeps levels 0 and 1 (approximation and first detail)
            
        Returns:
            Filtered time series
        """
        if keep_levels is None:
            keep_levels = [0, 1]  # Default: keep approximation and first detail level
        
        # Convert to numpy array
        data = series.values
        
        # Decompose
        coeffs = pywt.wavedec(data, self.wavelet, mode=self.mode, level=self.level)
        
        # Zero out coefficients for levels not in keep_levels
        new_coeffs = []
        for i in range(len(coeffs)):
            if i in keep_levels:
                new_coeffs.append(coeffs[i])
            else:
                new_coeffs.append(np.zeros_like(coeffs[i]))
        
        # Reconstruct signal
        filtered_data = pywt.waverec(new_coeffs, self.wavelet, mode=self.mode)
        
        # Trim to original length if needed
        if len(filtered_data) > len(series):
            filtered_data = filtered_data[:len(series)]
        
        return pd.Series(filtered_data, index=series.index) 
    



class EMDFilter:
    """
    Empirical Mode Decomposition filter for adaptive time series analysis.
    
    EMD decomposes a signal into Intrinsic Mode Functions (IMFs) which represent
    different oscillatory modes present in the data. It's especially useful for
    non-linear and non-stationary time series.
    """
    
    def __init__(self, max_imfs: int = 10):
        """
        Initialize EMD filter.
        
        Parameters:
            max_imfs: Maximum number of IMFs to extract
        """
        self.max_imfs = max_imfs
        self.emd = PyEMD()
    
    def decompose(self, series: pd.Series) -> Dict[str, pd.Series]:
        """
        Decompose time series into Intrinsic Mode Functions (IMFs).
        
        Parameters:
            series: Input time series
            
        Returns:
            Dictionary of IMF series
        """
        # Set max IMFs
        self.emd.MAX_ITERATION = 2000
        
        # Decompose
        imfs = self.emd(series.values, max_imf=self.max_imfs)
        
        # Create dictionary with IMFs
        result = {}
        for i in range(imfs.shape[0]):
            result[f'imf_{i+1}'] = pd.Series(imfs[i, :], index=series.index)
        
        return result
    
    def filter_series(self, series: pd.Series, keep_imfs: List[int] = None) -> pd.Series:
        """
        Filter time series by selecting specific IMFs for reconstruction.
        
        Parameters:
            series: Input time series
            keep_imfs: IMF indices to keep (1-based indexing)
                      If None, keeps all but the first IMF (trend)
            
        Returns:
            Filtered time series
        """
        # Decompose
        imfs = self.emd(series.values, max_imf=self.max_imfs)
        
        if keep_imfs is None:
            # Default: keep all IMFs except the first (which often captures the trend)
            keep_imfs = list(range(2, imfs.shape[0] + 1))
        
        # Adjust to 0-based indexing
        imf_indices = [i - 1 for i in keep_imfs if 1 <= i <= imfs.shape[0]]
        
        # Sum selected IMFs
        if imf_indices:
            filtered_data = np.sum(imfs[imf_indices, :], axis=0)
        else:
            filtered_data = np.zeros_like(series.values)
        
        return pd.Series(filtered_data, index=series.index)


class EnsembleFilter:
    """
    Ensemble filter that combines multiple filtering techniques.
    
    This filter applies multiple filtering methods to the same time series
    and combines their results using a weighted average or other ensemble methods.
    """
    
    def __init__(self, filters: List[Tuple[object, float]] = None):
        """
        Initialize ensemble filter.
        
        Parameters:
            filters: List of (filter_object, weight) tuples
        """
        self.filters = filters or []
    
    def add_filter(self, filter_obj, weight: float = 1.0):
        """
        Add a filter to the ensemble.
        
        Parameters:
            filter_obj: Filter object with filter_series method
            weight: Weight for this filter in the ensemble
        """
        self.filters.append((filter_obj, weight))
    
    def filter_series(self, series: pd.Series) -> pd.Series:
        """
        Apply ensemble filtering to time series.
        
        Parameters:
            series: Input time series
            
        Returns:
            Filtered time series
        """
        if not self.filters:
            return series
        
        # Apply each filter
        filtered_series = []
        weights = []
        
        for filter_obj, weight in self.filters:
            filtered = filter_obj.filter_series(series)
            filtered_series.append(filtered)
            weights.append(weight)
        
        # Normalize weights
        weights = np.array(weights) / sum(weights)
        
        # Combine results (weighted average)
        result = pd.Series(0.0, index=series.index)
        for i, filtered in enumerate(filtered_series):
            result += filtered * weights[i]
        
        return result

class CUSUMFilter:
    """
    CUSUM (Cumulative Sum) filter for detecting persistent changes in time series.
    
    The CUSUM filter is effective for detecting regime changes and structural breaks
    in financial time series by tracking cumulative deviations from the mean.
    """
    
    def __init__(self, threshold: Optional[Union[float, pd.Series]] = None,
                vol_lookback: int = 100, vol_mult: float = 1.0,
                return_direction: bool = False):
        """
        Initialize CUSUM filter.
        
        Parameters:
            threshold: Fixed threshold or None for dynamic volatility-based threshold
            vol_lookback: Lookback period for volatility calculation (if threshold is None)
            vol_mult: Multiplier for volatility threshold
            return_direction: Whether to return direction of changes (1 for positive, -1 for negative)
        """
        self.threshold = threshold
        self.vol_lookback = vol_lookback
        self.vol_mult = vol_mult
        self.return_direction = return_direction
    
    def _get_volatility(self, series: pd.Series) -> pd.Series:
        """Calculate rolling volatility based on returns."""
        # Calculate returns more safely to handle potential misalignment
        returns = series.pct_change().dropna()
        
        # Use rolling standard deviation of returns
        vol = returns.ewm(span=self.vol_lookback).std()
        return vol
    
    def filter_series(self, series: pd.Series) -> Union[pd.DatetimeIndex, pd.Series]:
        """
        Apply CUSUM filter to detect change points.
        
        Parameters:
            series: Input time series
            
        Returns:
            DatetimeIndex of detected change points if return_direction=False
            Series with direction values (1, -1) if return_direction=True
        """
        if self.threshold is None:
            # Use dynamic threshold based on volatility
            threshold = self._get_volatility(series) * self.vol_mult
        else:
            threshold = self.threshold
            
        tEvents = []
        tDirections = []
        sPos, sNeg = 0, 0
        diff = series.diff().fillna(0)
        
        for i in range(1, len(series)):
            idx = series.index[i]
            curr_diff = diff.iloc[i]
            
            # Get threshold value for current index
            if isinstance(threshold, pd.Series):
                # Try to get exact index match, or use the last available value
                if idx in threshold.index:
                    h = threshold.loc[idx]
                else:
                    # Get closest previous value
                    prev_idx = threshold.index[threshold.index.get_indexer([idx], method='pad')[0]]
                    h = threshold.loc[prev_idx]
            else:
                h = threshold
            
            sPos = max(0, sPos + curr_diff)
            sNeg = min(0, sNeg + curr_diff)
            
            if sNeg < -h:
                sNeg = 0
                tEvents.append(idx)
                tDirections.append(-1)  # Negative change
            elif sPos > h:
                sPos = 0
                tEvents.append(idx)
                tDirections.append(1)   # Positive change
        
        if self.return_direction:
            # Return Series with direction
            if not tEvents:
                return pd.Series(dtype=float)  # Empty series
            
            events = pd.Series(tDirections, index=tEvents)
            return events
        else:
            # Return DatetimeIndex without direction
            return pd.DatetimeIndex(tEvents)