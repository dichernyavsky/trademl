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
    """

    def __init__(self, params: dict | None = None, barrier_strategy=None):
        super().__init__(params, barrier_strategy)

        # ---- event‑generator ------------------------------------------------------
        gen_params = {
            "lookback": self.params.get("lookback", 20),
            "mode": self.params.get("mode", "breakout"),
            "include_replacement": self.params.get("include_replacement", "ignore"),
        }
        self.event_generator = SimpleSREventGenerator(**gen_params)

        # ---- strategy‑specific params --------------------------------------------
        self.entry_offset: int = int(self.params.get("entry_offset", 0))
        if self.entry_offset < 0:
            raise ValueError("entry_offset must be >= 0")

    # ------------------------------------------------------------------------- #
    # Internal API
    # ------------------------------------------------------------------------- #
    def _generate_raw_events(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate events with entry shifted by ``entry_offset`` bars.

        The implementation is fully vectorised: no explicit Python loops, so it
        scales linearly with the number of rows even for tens of thousands of
        events.

        If ``entry_offset`` pushes an entry past the last bar available, that
        event is silently dropped.
        """
        # 1) Signal‑bar events
        events = self.event_generator.generate(data, include_entry_price=True)

        # fast‑exit if no shift requested
        if self.entry_offset == 0 or events.empty:
            return events

        # 2) Compute new (delayed) index positions -------------------------------
        signal_pos = data.index.get_indexer(events.index)
        delayed_pos = signal_pos + self.entry_offset

        # keep only positions that are still inside the data range
        in_bounds = delayed_pos < len(data.index)
        if not np.any(in_bounds):
            # all delayed entries fall outside data ⇒ no tradable events
            return events.iloc[0:0]  # empty frame with same columns

        delayed_pos = delayed_pos[in_bounds]
        new_index = data.index[delayed_pos]

        # 3) Re‑index without losing the original per‑event information ----------
        events = events.iloc[in_bounds].copy()
        events.index = new_index  # move each row to its entry bar

        # 4) Re‑price entry to *close* of the entry bar
        events["entry_price"] = data.loc[new_index, "Close"].values

        return events.sort_index()

# Alias for backward compatibility
SimpleSREventStrategy = SimpleSRStrategy

