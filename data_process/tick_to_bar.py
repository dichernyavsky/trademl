from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Literal, Tuple, Optional, Dict, Any

# ========= helpers: UTC-naive =========
# --- 1) возьми твою функцию БЕЗ ИЗМЕНЕНИЙ ---
def label_candles_netflow(
    buy_qty: np.ndarray,
    sell_qty: np.ndarray,
    threshold: float,
    mode: Literal["abs","pos","neg"] = "abs",
    carry_over: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(buy_qty)
    ids = np.empty(n, dtype=np.int64)
    net = 0.0
    cid = 0
    closes = []
    thr = float(threshold)
    for i, (vb, vs) in enumerate(zip(buy_qty, sell_qty)):
        ids[i] = cid
        net += float(vb) - float(vs)
        hit = (
            (mode == "abs" and abs(net) >= thr) or
            (mode == "pos" and net >= thr) or
            (mode == "neg" and -net >= thr)
        )
        if hit:
            closes.append(i)
            if carry_over:
                if mode == "pos":
                    net -= thr
                elif mode == "neg":
                    net += thr
                else:
                    net -= thr * (1.0 if net >= 0 else -1.0)
            else:
                net = 0.0
            cid += 1
    return ids, np.asarray(closes, dtype=np.int64)

# --- 2) добавь аналог для объёма (BUY/SELL/EITHER) ---
def label_candles_volume_dualside(
    buy_qty: np.ndarray,
    sell_qty: np.ndarray,
    threshold: float,
    wait_for: Literal["buy","sell","either"] = "either",
    carry_over: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Размечает свечи по накопленному объёму.
    Гарантия: для ВСЕХ ПОЛНЫХ свечей выполнено:
      - 'buy'   : cum_buy >= threshold
      - 'sell'  : cum_sell >= threshold
      - 'either': max(cum_buy, cum_sell) >= threshold
    Возвращает: (ids, close_idx, close_dir), где close_dir = +1 (buy) / -1 (sell)
    """
    assert wait_for in ("buy", "sell", "either")
    n = len(buy_qty)
    ids = np.empty(n, dtype=np.int64)
    cid = 0
    thr = float(threshold)
    cum_b = 0.0
    cum_s = 0.0
    closes = []
    close_dir = []  # +1 buy, -1 sell

    for i, (vb, vs) in enumerate(zip(buy_qty, sell_qty)):
        ids[i] = cid
        cum_b += float(vb)
        cum_s += float(vs)

        hit = False
        direction = 0
        if wait_for == "buy":
            if cum_b >= thr:
                hit = True
                direction = +1
        elif wait_for == "sell":
            if cum_s >= thr:
                hit = True
                direction = -1
        else:  # either = total volume mode
            total = cum_b + cum_s
            if total >= thr:
                hit = True
                # направление — по преобладающей стороне
                direction = +1 if (cum_b - cum_s) >= 0 else -1

        if hit:
            closes.append(i)
            close_dir.append(direction)
            if carry_over:
                if direction == +1:
                    cum_b -= thr
                else:
                    cum_s -= thr
            else:
                cum_b = 0.0
                cum_s = 0.0
            cid += 1

    return ids, np.asarray(closes, dtype=np.int64), np.asarray(close_dir, dtype=np.int8)


# ========= основной класс =========
class TickBarBuilder:
    """
    Универсальный конвертер тиков в бары по порогам:
      - method='volume' : порог по BUY/SELL/EITHER объёму (qty или quote_qty)
      - method='netflow': порог по накопленному нетто-потоку (buy - sell)

    Ожидаемые колонки df_trades:
      ['time','price','qty','quote_qty','is_buyer_maker']
      time — epoch миллисекунды (UTC)
      is_buyer_maker: True => SELL, False => BUY
    """

    def __init__(self, method: Literal["volume","netflow"], threshold: float,
                 dollar_vol: bool = False,
                 wait_for: Literal["buy","sell","either"] = "either",
                 net_mode: Literal["abs","pos","neg"] = "abs",
                 include_last_partial: bool = False,
                 carry_over: bool = False):
        self.method = method
        self.threshold = float(threshold)
        self.dollar_vol = bool(dollar_vol)
        self.wait_for = wait_for
        self.net_mode = net_mode
        self.include_last_partial = include_last_partial
        self.carry_over = carry_over

    def _validate(self, df: pd.DataFrame):
        required = {"time", "price", "qty", "quote_qty", "is_buyer_maker"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

    def _prepare_streams(self, df: pd.DataFrame) -> Dict[str, Any]:
        base = "quote_qty" if self.dollar_vol else "qty"
        # BUY: ~is_buyer_maker; SELL: is_buyer_maker
        buy_series = np.where(~df["is_buyer_maker"].astype(bool), df[base].astype(float), 0.0)
        sell_series = np.where(df["is_buyer_maker"].astype(bool), df[base].astype(float), 0.0)
        times_raw = pd.Series(df["time"].to_numpy())  # epoch ms (int)
        return {"base": base, "buy": buy_series, "sell": sell_series, "times": times_raw}


    def build(self, df_trades: pd.DataFrame) -> pd.DataFrame:
        self._validate(df_trades)
        df = df_trades.copy()
        df.sort_values("time", inplace=True, kind="mergesort", ignore_index=True)

        base = "quote_qty" if self.dollar_vol else "qty"
        buy = np.where(~df["is_buyer_maker"].astype(bool), df[base].astype(float), 0.0)
        sell = np.where(df["is_buyer_maker"].astype(bool), df[base].astype(float), 0.0)

        # === ВАЖНО: вместо boundaries+digitize — поточная разметка ===
        if self.method == "netflow":
            candle_id, close_idx = label_candles_netflow(
                buy, sell, self.threshold, mode=self.net_mode, carry_over=self.carry_over
            )
            close_dir = None
        elif self.method == "volume":
            candle_id, close_idx, close_dir = label_candles_volume_dualside(
                buy, sell, self.threshold, wait_for=self.wait_for, carry_over=self.carry_over
            )
        else:
            raise ValueError("method must be 'volume' or 'netflow'")

        df["candle_id"] = candle_id
        df["time_dt"] = pd.to_datetime(df["time"], unit="ms", utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
        df["qty_buy"] = buy
        df["qty_sell"] = sell

        ohlcv = df.groupby("candle_id", as_index=False).agg(
            Open=("price", "first"),
            High=("price", "max"),
            Low=("price", "min"),
            Close=("price", "last"),
            Volume_buy=("qty_buy", "sum"),
            Volume_sell=("qty_sell", "sum"),
            Trades=("price", "size"),
            OpenTime=("time_dt", "first"),
            CloseTime=("time_dt", "last"),
        )

        # Полноценность свечей: есть ли закрывающий тик?
        #is_complete = np.zeros(len(ohlcv), dtype=bool)
        #if len(close_idx):
            # Какой candle_id у закрывающего тика i? Это просто candle_id[i] (в df)
            #last_close_cid = df.loc[close_idx, "candle_id"].to_numpy()
            #is_complete[last_close_cid] = True
        #ohlcv["IsComplete"] = is_complete

        # (опционально) направление закрытия для volume-режима
        #if (self.method == "volume") and (close_dir is not None) and len(close_dir):
            # close_dir сопоставляется только с полными свечами
            #tmp = pd.Series(index=np.where(is_complete)[0], data=close_dir[:np.sum(is_complete)], dtype="int8")
            #ohlcv["CloseDir"] = tmp.reindex(range(len(ohlcv))).to_numpy()

        ohlcv["Duration"] = (ohlcv["CloseTime"] - ohlcv["OpenTime"]).dt.total_seconds()
        ohlcv["Volume"] = ohlcv["Volume_buy"] + ohlcv["Volume_sell"]

        ohlcv.set_index("OpenTime", inplace=True)
        return ohlcv

# ======= Пример использования =======
# builder = TickBarBuilder(
#     method="volume",            # или "netflow"
#     threshold=5_000,            # порог (в qty или в quote_qty)
#     dollar_vol=False,           # False -> qty, True -> quote_qty
#     wait_for="either",          # для method="volume": 'buy'|'sell'|'either'
#     net_mode="abs",             # для method="netflow": 'abs'|'pos'|'neg'
#     include_last_partial=False  # включать ли незакрытый последний бар
# )
# bars = builder.build(df_trades)
