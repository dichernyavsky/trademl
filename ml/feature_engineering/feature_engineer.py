"""
Feature engineer responsible for generating and attaching indicator‑based
features to DataFrameTrades.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Union, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CORE_OHLCV_COLS: Set[str] = {"Open", "High", "Low", "Close", "Volume"}
TRADE_META_COLS: Set[str] = {
    "bin",
    "direction",
    "entry_price",
    "exit_price",
    "exit_time",
    "pt",
    "sl",
    "t1",
    "dynamic_stop_path",
}

EXCLUDE_COLS: Set[str] = CORE_OHLCV_COLS | TRADE_META_COLS


class FeatureEngineer:
    """Attach indicator features from *market_data* to *trades*.

    Parameters
    ----------
    base_manager : IndicatorManager | None
        Manager responsible for computing same‑timeframe indicators.
    htf_manager : HigherTimeframeIndicatorManager | None
        Manager responsible for computing higher‑timeframe indicators.
    auto_compute : bool, default ``True``
        Recalculate missing columns automatically if managers are
        provided.
    """

    def __init__(
        self,
        *,
        base_manager: Optional = None,
        htf_manager: Optional = None,
        auto_compute: bool = True,
    ) -> None:
        self.base_manager = base_manager
        self.htf_manager = htf_manager
        self.auto_compute = auto_compute

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def enrich(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        *,
        compute_missing: Optional[bool] = None,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Return *data* copy with base‑TF indicators appended."""
        compute = self._decide_compute(compute_missing)
        if not compute or self.base_manager is None:
            return data  # nothing to do

        return self.base_manager.calculate_all(data, append=True)

    def add_to_trades(
        self,
        *,
        trades: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        market_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        htf_data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
        columns: Optional[List[str]] = None,
        compute_missing: Optional[bool] = None,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Join indicator features with *trades* at entry‑times."""
        compute = self._decide_compute(compute_missing)

        if isinstance(trades, dict):
            out: Dict[str, pd.DataFrame] = {}
            for sym, tdf in trades.items():
                md = market_data[sym] if isinstance(market_data, dict) else market_data
                hd = htf_data[sym] if isinstance(htf_data, dict) else htf_data
                out[sym] = self._add_single(tdf, md, hd, columns, compute)
            return out

        if isinstance(trades, pd.DataFrame):
            return self._add_single(trades, market_data, htf_data, columns, compute)

        raise TypeError("trades must be DataFrame or dict of DataFrames")

    def get_feature_names(self, *, split: bool = False) -> Union[List[str], Tuple[List[str], List[str]]]:
        """Return names of features produced by both managers.

        Parameters
        ----------
        split : bool, default ``False``
            If ``True`` the method returns a tuple ``(base_cols,
            htf_cols)`` instead of a single merged list.
        """
        base_cols: List[str] = []
        htf_cols: List[str] = []

        if self.base_manager is not None and hasattr(self.base_manager, "get_indicator_columns"):
            base_cols = list(self.base_manager.get_indicator_columns())
        if self.htf_manager is not None and hasattr(self.htf_manager, "get_indicator_columns"):
            htf_cols = list(self.htf_manager.get_indicator_columns())

        if split:
            return base_cols, htf_cols
        return base_cols + htf_cols

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _add_single(
        self,
        trades_df: pd.DataFrame,
        market_df: pd.DataFrame,
        htf_df: Optional[pd.DataFrame],
        columns: Optional[Sequence[str]],
        compute: bool,
    ) -> pd.DataFrame:
        """Process a single symbol."""
        # Ensure base features are present
        if compute and self.base_manager is not None:
            missing = self._missing_columns(market_df, self.base_manager.get_indicator_columns())
            if missing:
                market_df = self.base_manager.calculate_all(market_df, append=True)

        # Ensure HTF features are present
        if compute and self.htf_manager is not None and htf_df is not None:
            missing_htf = self._missing_columns(htf_df, self.htf_manager.get_indicator_columns())
            if missing_htf:
                htf_df = self.htf_manager.calculate_all(
                    lower_timeframe_data=market_df,
                    higher_timeframe_data=htf_df,
                    append=True,
                )

        # Choose columns to join
        if columns is None:
            columns = [c for c in market_df.columns if c.lower() not in EXCLUDE_COLS]
        if not columns:
            return trades_df.copy()

        # Join base‑TF features
        entry_bar_ids = self._entry_bar_ids(trades_df)
        out = trades_df.join(market_df.loc[entry_bar_ids, columns], how="left")

        # Join HTF features
        if self.htf_manager is not None and htf_df is not None:
            htf_cols = [c for c in self.htf_manager.get_indicator_columns() if c in htf_df.columns]
            if htf_cols:
                out = out.join(htf_df.loc[entry_bar_ids, htf_cols], how="left")

        return out

    # ------------------------- static helpers -------------------------
    @staticmethod
    def _entry_bar_ids(trades_df: pd.DataFrame) -> pd.Index:
        if "UniqueBarID" in trades_df.columns:
            return pd.Index(trades_df["UniqueBarID"].values)
        raise ValueError("Cannot determine entry bar IDs – provide 'UniqueBarID' column.")

    @staticmethod
    def _missing_columns(df: pd.DataFrame, expected: Sequence[str]) -> List[str]:
        return [col for col in expected if col not in df.columns]

    def _decide_compute(self, flag: Optional[bool]) -> bool:
        """Resolve local flag versus global default."""
        return self.auto_compute if flag is None else bool(flag)
