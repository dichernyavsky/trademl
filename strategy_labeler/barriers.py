import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict, Union
from .base_barrier import Barrier

# ----- SIMPLE VOLATILITY BARRIER CLASS ------------------------------------------------------------------

class SimpleVolatilityBarrier(Barrier):
    """
    Simple volatility-based barrier strategy.
    Calculates PT/SL barriers based on price volatility.
    """
    def __init__(
        self,
        window: int = 20,
        multiplier: List[float] = [2.0, 2.0],
        min_ret: float = 0.001,
        hold_periods: int = 60,
        vol_method: str = "bar_ewm",
        vol_kwargs: Optional[Dict] = None,
    ):
        """
        Initialize SimpleVolatilityBarrier.

        Args:
            window (int): Lookback window for volatility calculation (used by some methods)
            multiplier (list): [pt_multiplier, sl_multiplier] for volatility scaling
            min_ret (float): Minimum return threshold for barriers
            hold_periods (int): Number of bars/candles to hold for vertical barrier
            vol_method (str): Which volatility estimator to use. Options:
                - "bar_ewm"          : EWMA per-bar volatility (default for volume/dollar bars)
                - "bar_volnorm_ewm"  : EWMA per-bar volatility with volume normalization
                - "std"              : rolling close-to-close standard deviation
                - "ewm_std"          : exponentially weighted std (span-based)
                - "mad"              : rolling Median Absolute Deviation, scaled to sigma
                - "winsor_std"       : rolling std after winsorizing returns
                - "biweight"         : Tukey’s biweight midvariance (robust rolling estimator)
                - "parkinson"        : range-based (uses High/Low)
                - "garman_klass"     : range-based (uses Open/High/Low/Close)
                - "rogers_satchell"  : range-based (uses Open/High/Low/Close)
                - "yang_zhang"       : range-based (uses Open/High/Low/Close)
            vol_kwargs (dict): additional kwargs to pass to the vol estimator
        """
        super().__init__(hold_periods=hold_periods, window=window, multiplier=multiplier, min_ret=min_ret)
        self.window = int(window)
        self.multiplier = multiplier
        self.min_ret = float(min_ret)
        self.vol_method = vol_method
        self.vol_kwargs = vol_kwargs or {}

    def _calculate_horizontal_barriers(self, events: pd.DataFrame, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Calculate horizontal barriers using volatility-based approach.
        Accounts for position direction (long/short).

        Args:
            events: DataFrame with events (must have 'direction' column)
            data: DataFrame with OHLCV data
            **kwargs: Additional parameters (can override instance variables)

        Returns:
            DataFrame: Events with added 'pt' and 'sl' columns
        """
        # Use kwargs if provided, otherwise use instance variables
        window     = kwargs.get('window', self.window)
        multiplier = kwargs.get('multiplier', self.multiplier)
        min_ret    = kwargs.get('min_ret', self.min_ret)
        vol_method = kwargs.get('vol_method', self.vol_method)
        vol_kwargs = {**self.vol_kwargs, **kwargs.get('vol_kwargs', {})}

        # ---- validation / normalization (vectorization-friendly) -----------------------
        if not isinstance(multiplier, (list, tuple, np.ndarray)) or len(multiplier) != 2:
            raise ValueError("multiplier must be a sequence [pt_multiplier, sl_multiplier] of length 2")
        m_pt, m_sl = float(multiplier[0]), float(multiplier[1])

        # Get close prices
        close = data['Close'].astype(float)

        # Calculate simple returns (some vol methods will reuse)
        rets = close.pct_change()

        # ---- compute volatility (pluggable) -------------------------------------------
        vol = compute_volatility(
            data=data,
            window=window,
            method=vol_method,
            returns=rets,
            **vol_kwargs,
        )

        # Get volatility at event times (index-align w/ events)
        event_vol = vol.reindex(events.index)

        # Calculate target sizes (maximum of volatility-based and minimum threshold)
        pt_target = np.maximum(m_pt * event_vol.values, min_ret)
        sl_target = np.maximum(m_sl * event_vol.values, min_ret)

        # Calculate barriers relative to entry price
        entry_prices = events['entry_price']
        result = events.copy()

        # ---- Vectorized barrier construction ------------------------------------------
        # Calculate barriers based on position direction
        # For longs: PT above entry, SL below entry
        # For shorts: PT below entry, SL above entry
        directions = result['direction'].to_numpy()
        entry = entry_prices.to_numpy(dtype=float)

        # Default (long/neutral) formulas
        pt_long_like = entry * (1.0 + pt_target)
        sl_long_like = entry * (1.0 - sl_target)

        # Short formulas
        pt_short = entry * (1.0 - pt_target)
        sl_short = entry * (1.0 + sl_target)

        # Masks
        short_mask = (directions == -1)
        # Neutral events use default long-like barriers
        # Combine via np.where (vectorized, no Python loops)
        pt = np.where(short_mask, pt_short, pt_long_like)
        sl = np.where(short_mask, sl_short, sl_long_like)

        # Assign back (preserve dtype as float)
        result.loc[:, 'pt'] = pt
        result.loc[:, 'sl'] = sl

        return result

# -------------------------- Volatility helpers ------------------------------------------------------------

def _winsorize_series(s: pd.Series, lower_q=0.01, upper_q=0.99) -> pd.Series:
    """Clip series by quantiles to reduce tail impact (robust-lite)."""
    lo, hi = s.quantile(lower_q), s.quantile(upper_q)
    return s.clip(lo, hi)


def compute_vol_bar_ewm(
    data: pd.DataFrame,
    span_bars: int = 50,
    winsor_q: Tuple[float, float] = (0.005, 0.995),
    use_abs: bool = False,
) -> pd.Series:
    """
    Bar-based EWMA volatility for volume/dollar bars (per-bar logic).
    - No time normalization (per-bar).
    - Light winsorization to stabilize tails.
    - Smooth either r^2 (variance) or |r| (robust alt).
    """
    close = data["Close"].astype(float)
    r = close.pct_change().fillna(0.0)

    # light winsorization
    lo, hi = r.quantile(winsor_q[0]), r.quantile(winsor_q[1])
    r = r.clip(lo, hi)

    stat = (r.abs() if use_abs else r**2)
    m = stat.ewm(span=span_bars, adjust=False).mean()
    sigma = np.sqrt(m).fillna(0.0)
    return sigma


def compute_vol_bar_volume_norm_ewm(
    data: pd.DataFrame,
    span_bars: int = 50,
    winsor_q: Tuple[float, float] = (0.005, 0.995),
    vol_col: str = "Volume",
    ref_volume: str = "median",  # "median" | "ewm"
    ref_span: int = 200,         # span for ewm reference if ref_volume="ewm"
) -> pd.Series:
    """
    EWMA volatility with volume normalization (Mixture-of-Distributions flavor).
    - r_norm = r / sqrt(max(eps, V_t / V_ref))
    - Stabilizes variance if actual bar volumes vary around a target.
    """
    close = data["Close"].astype(float)
    V = data[vol_col].astype(float)

    r = close.pct_change().fillna(0.0)

    # Reference volume (typical bar volume)
    if ref_volume == "ewm":
        V_ref = V.ewm(span=ref_span, adjust=False).mean().clip(lower=1e-12)
    else:
        V_ref = pd.Series(V.median(), index=V.index).clip(lower=1e-12)

    scale = np.sqrt((V / V_ref).clip(lower=1e-12))
    r_norm = (r / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # winsorize normalized returns
    lo, hi = r_norm.quantile(winsor_q[0]), r_norm.quantile(winsor_q[1])
    r_norm = r_norm.clip(lo, hi)

    m = (r_norm**2).ewm(span=span_bars, adjust=False).mean()
    sigma = np.sqrt(m).fillna(0.0)
    return sigma


def compute_volatility(
    data: pd.DataFrame,
    window: int = 20,
    method: str = "bar_ewm",
    returns: Optional[pd.Series] = None,
    # kwargs for specific methods:
    ewm_span: int = 20,
    trim_q: Tuple[float, float] = (0.01, 0.99),
    # bar methods:
    span_bars: int = 50,
    winsor_q: Tuple[float, float] = (0.005, 0.995),
    use_abs: bool = False,
    vol_col: str = "Volume",
    ref_volume: str = "median",
    ref_span: int = 200,
) -> pd.Series:
    """
    Compute volatility using various estimators (vectorized).

    Args:
        data: DataFrame with at least 'Close' (and for range-based: 'Open','High','Low').
        window: rolling window size (bars) for rolling methods.
        method: one of:
            - "bar_ewm": EWMA per-bar volatility (recommended default for volume-bars)
            - "bar_volnorm_ewm": EWMA per-bar volatility with volume normalization
            - "std": rolling close-to-close std
            - "ewm_std": exponentially weighted std (span = ewm_span)
            - "mad": rolling MAD scaled to sigma (1.4826 * MAD)
            - "winsor_std": rolling std after winsorizing returns by trim_q
            - "biweight": Tukey's biweight midvariance (rolling)
            - "parkinson": range-based estimator (needs High/Low)
            - "garman_klass": range-based (Open,High,Low,Close)
            - "rogers_satchell": range-based (Open,High,Low,Close)
            - "yang_zhang": range-based (Open,High,Low,Close)
        returns: optional precomputed returns Series; if None uses pct_change on Close.
        ewm_span, trim_q: params for respective methods.
        span_bars, winsor_q, use_abs, vol_col, ref_volume, ref_span: params for bar_* methods.

    Returns:
        pd.Series of volatility (sigma) aligned to data.index.
    """
    method = method.lower()

    # Volume-bar oriented defaults
    if method == "bar_ewm":
        return compute_vol_bar_ewm(
            data=data, span_bars=span_bars, winsor_q=winsor_q, use_abs=use_abs
        )

    if method == "bar_volnorm_ewm":
        return compute_vol_bar_volume_norm_ewm(
            data=data,
            span_bars=span_bars,
            winsor_q=winsor_q,
            vol_col=vol_col,
            ref_volume=ref_volume,
            ref_span=ref_span,
        )

    # Close-to-close returns (generic path)
    close = data["Close"].astype(float)
    rets = close.pct_change() if returns is None else returns
    rets = rets.astype(float)

    if method == "std":
        vol = rets.rolling(window).std()

    elif method == "ewm_std":
        vol = rets.ewm(span=ewm_span, adjust=False).std()

    elif method == "mad":
        # Rolling median absolute deviation scaled by 1.4826 to estimate sigma
        med = rets.rolling(window).median()
        mad = (rets - med).abs().rolling(window).median()
        vol = 1.4826 * mad

    elif method == "winsor_std":
        wr = _winsorize_series(rets, *trim_q)
        vol = wr.rolling(window).std()

    elif method == "biweight":
        # Tukey's biweight midvariance (robust).
        c = 9.0
        def biweight_std(x: Union[np.ndarray, pd.Series]) -> float:
            x = np.asarray(x)
            if len(x) == 0:
                return 0.0
            m = np.median(x)
            mad = np.median(np.abs(x - m))
            s = 1.4826 * mad + 1e-12
            u = (x - m) / (c * s)
            w = (1 - u**2)**2
            w[np.abs(u) >= 1] = 0.0
            num = np.sum(((x - m)**2) * w)
            den = (np.sum(w) - 1e-12)
            v = num / max(den, 1e-12)
            return float(np.sqrt(max(v, 0.0)))
        vol = rets.rolling(window).apply(biweight_std, raw=False)

    elif method in {"parkinson", "garman_klass", "rogers_satchell", "yang_zhang"}:
        required = {"Open", "High", "Low", "Close"}
        if not required.issubset(data.columns):
            raise ValueError(f"{method} requires Open/High/Low/Close in data")

        O, H, L, C = (
            data["Open"].astype(float),
            data["High"].astype(float),
            data["Low"].astype(float),
            data["Close"].astype(float),
        )

        if method == "parkinson":
            # sigma^2 = (1/(4 ln2)) * mean( (ln(H/L))^2 ), then sqrt
            rs = (np.log(H / L)) ** 2
            vol = np.sqrt((rs.rolling(window).mean()) / (4.0 * np.log(2.0)))

        elif method == "garman_klass":
            log_hl = np.log(H / L)
            log_co = np.log(C / O)
            var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
            vol = np.sqrt(var.rolling(window).mean().clip(lower=0))

        elif method == "rogers_satchell":
            var = (np.log(H / O) * np.log(H / C) + np.log(L / O) * np.log(L / C))
            vol = np.sqrt(var.rolling(window).mean().clip(lower=0))

        elif method == "yang_zhang":
            k = 0.34
            log_oo = np.log(O.shift(1) / O)
            log_oc = np.log(C / O)
            log_hl = np.log(H / L)

            sigma_o2 = (log_oo**2).rolling(window).mean()
            sigma_c2 = (log_oc**2).rolling(window).mean()
            sigma_rs = (0.5 * log_hl**2 - (2 * np.log(2) - 1) * (log_oc**2)).rolling(window).mean()

            var = sigma_o2 + k * sigma_c2 + (1 - k) * sigma_rs
            vol = np.sqrt(var.clip(lower=0))
    else:
        raise ValueError(f"Unknown volatility method: {method}")

    return vol


