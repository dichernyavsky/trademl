import numpy as np
import pandas as pd

from .base_indicator import BaseIndicator


class RSIIndicator(BaseIndicator):
    """
    Relative Strength Index (RSI).

    Calculates classic RSI and its [0, 1] normalized version.
    """

    def __init__(self, window: int = 14, column: str = "Close"):
        self.window = window
        self.column = column
        super().__init__()

    # ------------------------------------------------------------------
    # Helper: column names
    # ------------------------------------------------------------------
    def get_column_names(self, **kwargs):
        return [
            f"RSI_{self.column}_{self.window}",
            f"RSI_{self.column}_{self.window}_Norm",  # normalized to 0-1
        ]

    # ------------------------------------------------------------------
    # Core calculation
    # ------------------------------------------------------------------
    def _calculate_for_single_df(self, data: pd.DataFrame, append: bool = True, **kwargs):
        if self.column not in data.columns:
            raise ValueError(f"Column '{self.column}' not found in data")

        delta = data[self.column].diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)

        # Use Wilder's smoothing (EMA-like) for average gains/losses
        avg_gain = pd.Series(gain, index=data.index).ewm(alpha=1 / self.window, adjust=False).mean()
        avg_loss = pd.Series(loss, index=data.index).ewm(alpha=1 / self.window, adjust=False).mean()

        rs = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - 100 / (1 + rs)
        rsi_norm = rsi / 100.0  # scale to [0, 1]

        indicator_data = {
            self.column_names[0]: rsi,
            self.column_names[1]: rsi_norm,
        }
        self.values = indicator_data
        self.is_calculated = True
        return self._append_to_df(data, indicator_data) if append else self._create_indicator_df(data, indicator_data)


class MACDIndicator(BaseIndicator):
    """
    Moving Average Convergence Divergence (MACD).

    Produces MACD line, Signal line and Histogram.
    All values are dimensionless as they are calculated from price differences.
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, column: str = "Close"):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.column = column
        super().__init__()

    # ------------------------------------------------------------------
    def get_column_names(self, **kwargs):
        base = f"MACD_{self.column}_{self.fast}_{self.slow}_{self.signal}"
        return [
            f"{base}_Line",
            f"{base}_Signal",
            f"{base}_Hist",
        ]

    # ------------------------------------------------------------------
    def _calculate_for_single_df(self, data: pd.DataFrame, append: bool = True, **kwargs):
        if self.column not in data.columns:
            raise ValueError(f"Column '{self.column}' not found in data")

        ema_fast = data[self.column].ewm(span=self.fast, adjust=False).mean()
        ema_slow = data[self.column].ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
        hist = macd_line - signal_line

        # Optional: normalize histogram to [-1, 1] by tanh to stabilize scale (commented)
        # hist = np.tanh(hist / (np.std(hist.dropna()) + 1e-9))

        indicator_data = {
            self.column_names[0]: macd_line,
            self.column_names[1]: signal_line,
            self.column_names[2]: hist,
        }

        self.values = indicator_data
        self.is_calculated = True
        return self._append_to_df(data, indicator_data) if append else self._create_indicator_df(data, indicator_data) 