from __future__ import annotations

from .base_strategy import BaseStrategy
# NOTE: changed for local testing in Jupyter
from ..events.indicator_events import SimpleSREventGenerator
#from events.indicator_events import SimpleSREventGenerator
import pandas as pd
import numpy as np


class SimpleSRStrategy(BaseStrategy):
    """SR‑breakout strategy with optional delayed entry.

    Parameters expected in ``params`` dict
    --------------------------------------
    lookback : int, default 20
        Window length for SR calculation (passed straight to the event generator).
    mode : {"breakout", "touch"}, default "breakout"
        How the event generator defines the SR interaction.
    include_replacement : {"ignore", "replace", "parallel"}, default "ignore"
        How the generator handles several events on the same bar.
    entry_offset : int, default 0
        **NEW.** Number of bars to wait *after* the signal candle before entering.
        ``0`` keeps the original behaviour (entry on signal bar), ``1`` delays by
        one full candle, etc.
    entry_price_mode : {"close", "breakout"}, default "close"
        **NEW.** Price to use for entry:
        - "close": Use the close price of the entry bar (after offset)
        - "breakout": Use the original breakout price (resistance/support level)
    """

    def __init__(self, params: dict | None = None, barrier_strategy=None):
        super().__init__(params, barrier_strategy)

        # ---- event‑generator ------------------------------------------------------
        gen_params = {
            "lookback": self.params.get("lookback", 20),
            "mode": self.params.get("mode", "breakout")
        }
        self.event_generator = SimpleSREventGenerator(**gen_params)
        
        self.entry_price_mode: str = self.params.get("entry_price_mode", "close")
        if self.entry_price_mode not in ("close", "breakout"):
            raise ValueError("entry_price_mode must be 'close' or 'breakout'")

    # ------------------------------------------------------------------------- #
    # Internal API
    # ------------------------------------------------------------------------- #
    def _generate_raw_events(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate events with entry at the same bar as the signal.

        The implementation is fully vectorised: no explicit Python loops, so it
        scales linearly with the number of rows even for tens of thousands of
        events.
        """
        # 1) Signal‑bar events
        events = self.event_generator.generate(data, include_entry_price=True)

        if events.empty:
            return events

        # 3) Set entry price based on mode ----------------------------------------
        if self.entry_price_mode == "close":
            # Use close price by UniqueBarID
            events["entry_price"] = data.loc[events['UniqueBarID'], 'Close'].values
        elif self.entry_price_mode == "breakout":
            # Keep the original breakout price (already calculated by event generator)
            # The entry_price column already contains the breakout prices
            pass  # No change needed, breakout prices are already in entry_price

        return events

# Alias for backward compatibility
SimpleSREventStrategy = SimpleSRStrategy
