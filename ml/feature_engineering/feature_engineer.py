"""
Feature engineer responsible for generating and attaching indicator‑based
features to DataFrameTrades.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Union

import pandas as pd

# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #
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
    """Мини‑engineer для индикаторных признаков.

    Parameters
    ----------
    indicators
        Iterable of indicator instances supporting ``calculate(data, append=True)``.
    htf_manager
        Optional manager, умеющий считать индикаторы на более высоком тайм‑фрейме.
    """

    def __init__(self, indicators: Optional[Sequence] = None, htf_manager: Optional = None):
        self.indicators: list = list(indicators) if indicators else []
        self.htf_manager = htf_manager

    # ---------------------------------------------------------------------
    # Public API                                                           |
    # ---------------------------------------------------------------------
    def enrich(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]):
        """Возвращает копию *data* с добавленными колонками индикаторов."""
        if not self.indicators:
            return data  # nothing to do

        if isinstance(data, pd.DataFrame):
            return self._enrich_single_df(data)
        elif isinstance(data, dict):
            return {sym: self._enrich_single_df(df) for sym, df in data.items()}
        else:
            raise TypeError("data must be a DataFrame or dict of DataFrames")

    def add_to_trades(
        self,
        trades: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        market_data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        columns: Optional[List[str]] = None,
        *,
        htf_data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Присоединяет *columns* с признаками к *trades* на момент входа.

        Возвращает **новый объект**, входные данные не мутируются.
        """
        if isinstance(trades, pd.DataFrame):
            return self._add_to_single_trades(trades, market_data, columns, htf_data)

        if isinstance(trades, dict):
            if not isinstance(market_data, (pd.DataFrame, dict)):
                raise TypeError("market_data must mirror trades structure")
            if htf_data is not None and not isinstance(htf_data, (pd.DataFrame, dict)):
                raise TypeError("htf_data must mirror trades structure")

            result: Dict[str, pd.DataFrame] = {}
            for sym, tdf in trades.items():
                md = market_data[sym] if isinstance(market_data, dict) else market_data
                hd = htf_data[sym] if isinstance(htf_data, dict) else htf_data
                result[sym] = self._add_to_single_trades(tdf, md, columns, hd)
            return result

        raise TypeError("trades must be a DataFrame or dict of DataFrames")

    # ------------------------------------------------------------------
    # Internal helpers                                                 |
    # ------------------------------------------------------------------
    def _enrich_single_df(self, df: pd.DataFrame) -> pd.DataFrame:
        enriched = df.copy()
        for ind in self.indicators:
            enriched = ind.calculate(enriched, append=True)
        return enriched

    def _add_to_single_trades(
        self,
        trades_df: pd.DataFrame,
        market_df: pd.DataFrame,
        columns: Optional[List[str]],
        htf_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        # 1. Определяем entry‑times
        entry_idx = self._get_entry_times(trades_df)

        # 2. Выбираем набор колонок для джоина
        if columns is None:
            columns = [c for c in market_df.columns if c.lower() not in EXCLUDE_COLS]
        if not columns:
            return trades_df.copy()  # nothing to join

        # 3. Джойним market‑фичи
        out = trades_df.join(market_df.loc[entry_idx, columns], how="left")

        # 4. (опционально) добавляем HTF‑признаки
        if self.htf_manager is not None and htf_df is not None:
            htf_full = self.htf_manager.calculate_all(market_df, htf_df, append=True)
            htf_cols = self.htf_manager.get_indicator_columns()
            out = out.join(htf_full.loc[entry_idx, htf_cols], how="left")

        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _get_entry_times(trades_df: pd.DataFrame) -> pd.Index:
        """Возвращает ``pd.Index`` моментов входа."""
        if "t0" in trades_df.columns:
            return pd.Index(trades_df["t0"].values)
        if isinstance(trades_df.index, pd.DatetimeIndex):
            return trades_df.index
        raise ValueError("Cannot determine entry times – provide 't0' column or DatetimeIndex.")