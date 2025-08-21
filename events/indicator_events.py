"""
Event generators based on technical indicators.

These event generators use custom indicator classes to generate directional trade events.
"""
from __future__ import annotations

from typing import Literal, Dict, Any, Optional
import pandas as pd
import numpy as np
from .base import EventGenerator, EventType
from ..indicators.volatility import *
from ..indicators.volume import *
from ..indicators.support_resistance import *

class BollingerBandsEventGenerator(EventGenerator):
    """
    Generates events based on Bollinger Bands indicator.
    
    Events are generated when price crosses or touches the bands, with
    expected direction based on mean reversion or breakout.
    """
    
    def __init__(self, window: int = 20, num_std: float = 2.0, 
                 mode: str = "reversal"):
        """
        Initialize the Bollinger Bands event generator.
        
        Parameters:
            window: Window size for moving average calculation
            num_std: Number of standard deviations for the bands
            mode: 'reversal' for mean reversion, 'breakout' for momentum strategy
        """
        super().__init__(event_type=EventType.DIRECTION_SPECIFIC)
        self.window = window
        self.num_std = num_std
        self.mode = mode
        
        # Add Bollinger Bands indicator
        self.add_indicator(BollingerBands(window=window, num_std=num_std))
    
    def generate(self, data: pd.DataFrame) -> pd.DataFrame:
        indicator_results = self.calculate_indicators(data)
        middle_band = indicator_results['Bollinger_Bands_Middle']
        upper_band = indicator_results['Bollinger_Bands_Upper']
        lower_band = indicator_results['Bollinger_Bands_Lower']
        
        # Get close prices for signal generation
        self.close = data['Close']
        
        # Initialize events
        events = pd.Series(0, index=data.index)
        
        # Mean reversion strategy: buy when price hits lower band, sell when it hits upper band
        upper_touch = self.close >= upper_band
        lower_touch = self.close <= lower_band
        if self.mode == "reversal":
            # Set event directions
            events[upper_touch] = -1  # Sell signal - expect move down toward mean
            events[lower_touch] = 1   # Buy signal - expect move up toward mean
            
        elif self.mode == "breakout":
            # Breakout strategy: buy when price breaks above upper band, sell when it breaks below lower band
            events[upper_touch] = 1   # Buy signal - expect continued upward momentum
            events[lower_touch] = -1  # Sell signal - expect continued downward momentum
        
        # Keep only non-zero events
        events = events[events != 0]
        events_df = pd.DataFrame({'direction': events})
        
        self.events = events_df
        return self.events

# ---------------------------------------------------------------------
# Simple Support/Resistance Event Generator
# ---------------------------------------------------------------------

_InclRepl = Literal["ignore", "include", "only"]


def _normalise_flag(flag: str | None) -> _InclRepl:
    if flag is None:
        return "ignore"
    f = flag.lower()
    if f not in ("ignore", "include", "only"):
        raise ValueError(
            "include_replacement must be 'ignore', 'include' or 'only' – got %s" % flag
        )
    return f  # type: ignore [return-value]


# ---------------------------------------------------------------------------
# Event‑generator
# ---------------------------------------------------------------------------
class SimpleSREventGenerator(EventGenerator):
    """Generate breakout events on Support/Resistance levels."""

    def __init__(
        self,
        lookback: int = 20,
        mode: str = "breakout",
        *,
        include_replacement: _InclRepl | str = "only",
    ) -> None:
        super().__init__(event_type=EventType.DIRECTION_SPECIFIC)
        if mode != "breakout":
            raise ValueError("Only 'breakout' mode is supported right now.")

        self.lookback = int(lookback)
        self.mode = mode
        self.include_replacement: _InclRepl = _normalise_flag(include_replacement)

        # register indicator – calculated once per .generate() call
        self.add_indicator(SimpleSupportResistance(lookback=self.lookback))

    # ------------------------------------------------------------------
    def _detect_events(self, df: pd.DataFrame) -> pd.Series:
        """Return a *direction* Series (+1/‑1) with index == df.index."""

        res_col = f"SimpleSR_{self.lookback}_Resistance"
        sup_col = f"SimpleSR_{self.lookback}_Support"
        if res_col not in df.columns or sup_col not in df.columns:
            raise KeyError("Support/Resistance columns are missing – did you run the indicator?")

        high = df["High"]
        low = df["Low"]
        res = df[res_col]
        sup = df[sup_col]

        prev_res = res.shift(1)
        prev_sup = sup.shift(1)

        # ---------------- classic break‑out (line disappears) -------------
        res_end = prev_res.notna() & res.isna() & (high >= prev_res)  # resistance broken
        sup_end = prev_sup.notna() & sup.isna() & (low <= prev_sup)   # support broken

        # ---------------- replacement‑bar break‑out -----------------------
        res_repl = prev_res.notna() & res.notna() & (res != prev_res) & (high >= prev_res)
        sup_repl = prev_sup.notna() & sup.notna() & (sup != prev_sup) & (low <= prev_sup)

        # ---------------- combine according to flag -----------------------
        if self.include_replacement == "ignore":
            res_mask, sup_mask = res_end, sup_end
        elif self.include_replacement == "only":
            res_mask, sup_mask = res_repl, sup_repl
        else:  # "include"
            res_mask = res_end | res_repl
            sup_mask = sup_end | sup_repl

        direction = pd.Series(0, index=df.index, dtype="int8")
        direction.loc[res_mask] = +1
        direction.loc[sup_mask] = -1
        return direction

    # ------------------------------------------------------------------
    def generate(
        self,
        data: pd.DataFrame,
        *,
        include_entry_price: bool = False,
        keep_indicators: list[str] | None = None,
    ) -> pd.DataFrame:  # noqa: D401 – one‑liner OK here
        """Generate events dataframe."""

        keep_indicators = keep_indicators or []

        # ---- 1) indicator ----------------------------------------------------
        ind_results = self.calculate_indicators(data)
        df = data.join(ind_results, rsuffix="_ind")  # avoid column clashes

        # ---- 2) detect events -----------------------------------------------
        direction = self._detect_events(df)
        mask = direction != 0

        events = pd.DataFrame({"direction": direction[mask]})

        # ---- 3) entry price --------------------------------------------------
        if include_entry_price:
            res_col = f"SimpleSR_{self.lookback}_Resistance"
            sup_col = f"SimpleSR_{self.lookback}_Support"

            prev_res = df[res_col].shift(1)
            prev_sup = df[sup_col].shift(1)

            entry_price = pd.Series(index=df.index, dtype="float64")
            entry_price.loc[direction == +1] = prev_res[direction == +1]
            entry_price.loc[direction == -1] = prev_sup[direction == -1]

            events["entry_price"] = entry_price[mask]

        # ---- 4) debug: keep selected indicator columns ----------------------
        if keep_indicators:
            events = events.join(ind_results.loc[mask, keep_indicators])

        self.events = events
        return events