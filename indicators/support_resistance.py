from __future__ import annotations
import numpy as np
import pandas as pd
from .base_indicator import BaseIndicator



class SimpleSupportResistance(BaseIndicator):
    """
    Simple Support and Resistance indicator that identifies pivot points
    and extends them as support/resistance levels until broken.
    - activation after right lookback (shift)
    - strict breakout (res: High > level; sup: Low < level)
    - optional: "touch" = entering a narrow tick-based band near the level
    """

    def __init__(
        self,
        lookback: int = 20,
        high_col: str = "High",
        low_col: str = "Low",
        *,
        atol: float | None = None,         # допуск для строгих сравнений; None -> авто (≈ 1e-6 * median price)
        compute_band_touches: bool = True,
        band_ticks: float = 2.0,           # ширина полосы в тиках
        tick_size: float = 0.01            # ЕДИНЫЙ тик для всего df
    ):
        self.lookback = int(lookback)
        self.high_col = high_col
        self.low_col = low_col
        self.atol = atol

        self.compute_band_touches = bool(compute_band_touches)
        self.band_ticks = float(band_ticks)
        self.tick_size = float(tick_size)

        super().__init__()

    # === Public API expected by your infra ===
    def get_column_names(self):
        """Return column names produced by this indicator (only levels, pivots are internal)."""
        lb = self.lookback
        return [
            f"SimpleSR_{lb}_Resistance",
            f"SimpleSR_{lb}_Support",
        ]

    def _calculate_for_single_df(self, data: pd.DataFrame, append=True, **kwargs) -> pd.DataFrame:
        # allow runtime overrides
        lookback = int(kwargs.get("lookback", self.lookback))
        high_col = kwargs.get("high_col", self.high_col)
        low_col = kwargs.get("low_col", self.low_col)
        atol = kwargs.get("atol", self.atol)
        compute_band_touches = bool(kwargs.get("compute_band_touches", self.compute_band_touches))
        band_ticks = float(kwargs.get("band_ticks", self.band_ticks))
        tick_size = float(kwargs.get("tick_size", self.tick_size))

        df = data.copy()

        high = df[high_col].astype("float64")
        low = df[low_col].astype("float64")
        # auto atol (стабильно для разных масштабов)
        if atol is None:
            atol = float(10e-9)

        # ---------- 1) pivots (plateau-friendly) ----------
        ph = self._find_pivot_highs(df, lookback, high_col)
        pl = self._find_pivot_lows(df, lookback, low_col)

        # ---------- 2) activate after right-lookback (no look-ahead leak) ----------
        df["_PH_ACTIVE"] = ph#.shift(lookback)
        df["_PL_ACTIVE"] = pl#.shift(lookback)

        # ---------- 3) extend lines with strict breakout ----------
        res = self._extend_pivot_high_line(df, high_col, atol)   # uses _PH_ACTIVE
        sup = self._extend_pivot_low_line(df, low_col, atol)     # uses _PL_ACTIVE

        # names as required by your downstream code
        lb = lookback
        out = {
            f"SimpleSR_{lb}_Resistance": res,
            f"SimpleSR_{lb}_Support": sup,
        }

        # ---------- 4) optional band touches (vectorized) ----------
        if compute_band_touches:
            out.update(self._compute_band_touches(
                df=df,
                res=res,
                sup=sup,
                high=high,
                low=low,
                lookback=lb,
                band_ticks=band_ticks,
                tick_size=tick_size,
            ))

        # clean up temp columns
        df.drop(columns=[c for c in ["_PH_ACTIVE", "_PL_ACTIVE"] if c in df.columns], inplace=True, errors="ignore")

        self.is_calculated = True
        return self._append_to_df(data, out) if append else self._create_indicator_df(data, out)

    # === internals ===
    def _find_pivot_highs(self, df: pd.DataFrame, lookback: int, high_col: str) -> pd.Series:
        """
        Pivot high at i iff High[i] >= (>) max(High[i-L:i]) AND >= (>) max(High[i+1:i+1+L]).
        """
        high_values = df[high_col].astype("float64")
        left_window_max = high_values.shift(1).rolling(window=lookback, min_periods=lookback).max()
        right_window_max = high_values[::-1].shift(1).rolling(window=lookback, min_periods=lookback).max()[::-1]
        cond = (high_values >= left_window_max) & (high_values >= right_window_max)
        return high_values.where(cond)

    def _find_pivot_lows(self, df: pd.DataFrame, lookback: int, low_col: str) -> pd.Series:
        """
        Pivot low at i iff Low[i] <= (<) min(Low[i-L:i]) AND <= (<) min(Low[i+1:i+1+L]).
        """
        low_values = df[low_col].astype("float64")
        left_window_min = low_values.shift(1).rolling(window=lookback, min_periods=lookback).min()
        right_window_min = low_values[::-1].shift(1).rolling(window=lookback, min_periods=lookback).min()[::-1]
        cond = (low_values <= left_window_min) & (low_values <= right_window_min)
        return low_values.where(cond)

    def _extend_pivot_high_line(self, df: pd.DataFrame, high_col: str, atol: float) -> pd.Series:
        """
        Extend resistance from each activated pivot high (_PH_ACTIVE) forward
        until STRICT breakout (High > level + atol). Equality is NOT a breakout.
        """
        # groups start where _PH_ACTIVE is not NaN
        grp = df["_PH_ACTIVE"].notna().cumsum()
        # baseline is the pivot value itself (from _PH_ACTIVE), not the bar's High
        baseline = df.groupby(grp)["_PH_ACTIVE"].transform("first")
        # within-group position
        is_first = df.groupby(grp).cumcount() == 0
        # strict breakout (exclude first bar in group from violation check)
        violation = (~is_first) & (df[high_col].astype("float64") > baseline + atol)
        broken = violation.groupby(grp).cummax()
        # line persists until broken; NaN where no active pivot
        #line = baseline.where(~broken)
        # ensure we don't emit lines for non-active groups (where _PH_ACTIVE is NaN)
        #line = line.where(df["_PH_ACTIVE"].notna())
        line = baseline.where(~broken, np.nan)
        return line

    def _extend_pivot_low_line(self, df: pd.DataFrame, low_col: str, atol: float) -> pd.Series:
        """
        Extend support from each activated pivot low (_PL_ACTIVE) forward
        until STRICT breakout (Low < level - atol). Equality is NOT a breakout.
        """
        grp = df["_PL_ACTIVE"].notna().cumsum()
        baseline = df.groupby(grp)["_PL_ACTIVE"].transform("first")
        is_first = df.groupby(grp).cumcount() == 0
        violation = (~is_first) & (df[low_col].astype("float64") < baseline - atol)
        broken = violation.groupby(grp).cummax()
        #line = baseline.where(~broken)
        line = baseline.where(~broken, np.nan)
        return line

    def _compute_band_touches(
        self,
        *,
        df: pd.DataFrame,
        res: pd.Series,
        sup: pd.Series,
        high: pd.Series,
        low: pd.Series,
        lookback: int,
        band_ticks: float,
        tick_size: float,
    ) -> dict[str, pd.Series]:
        """
        Возвращает dict с колонками:
        SimpleSR_{lookback}_ResBandHit, SimpleSR_{lookback}_SupBandHit,
        SimpleSR_{lookback}_ResBandHits, SimpleSR_{lookback}_SupBandHits
        Логика: one-sided полоса в тиках, без засчёта бара пробоя.
        """
        lb = int(lookback)
        w = float(band_ticks) * float(tick_size)

        res_active = res.notna()
        sup_active = sup.notna()

        # one-sided bands
        res_lo = res - w
        res_hi = res
        sup_lo = sup
        sup_hi = sup + w

        # hit = пересечение [Low, High] с полосой + запрет пробоя на баре
        res_band_hit = res_active & (high >= res_lo) & (low <= res_hi) & (high <= res)
        sup_band_hit = sup_active & (high >= sup_lo) & (low <= sup_hi) & (low >= sup)

        # сегменты: активен и значение уровня неизменно
        res_seg_break = (~res_active) | (res.ne(res.shift(1)))
        sup_seg_break = (~sup_active) | (sup.ne(sup.shift(1)))
        res_seg_id = res_seg_break.cumsum().where(res_active)
        sup_seg_id = sup_seg_break.cumsum().where(sup_active)

        res_band_hits = res_band_hit.groupby(res_seg_id, dropna=True).cumsum().astype("Int64").where(res_active, 0)
        sup_band_hits = sup_band_hit.groupby(sup_seg_id, dropna=True).cumsum().astype("Int64").where(sup_active, 0)

        return {
            f"SimpleSR_{lb}_ResBandHit": res_band_hit,
            f"SimpleSR_{lb}_SupBandHit": sup_band_hit,
            f"SimpleSR_{lb}_ResBandHits": res_band_hits,
            f"SimpleSR_{lb}_SupBandHits": sup_band_hits,
        }
