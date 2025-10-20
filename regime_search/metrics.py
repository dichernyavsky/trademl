# metrics.py
from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any


# ========= Performance metrics =========

class SharpePerf:
    """
    perf_value = mean(R) / (std(R) + eps)

    Без risk-free: в наших внутрипроцессных/внутридневных сравнениях
    важна форма распределения, а не excess-return к бенчмарку.
    """
    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def __call__(self, R: np.ndarray, T: Optional[np.ndarray] = None) -> float:
        R = np.asarray(R, dtype=float)
        n = R.size
        if n == 0:
            return 0.0
        mu = float(np.mean(R))
        sigma = float(np.std(R, ddof=1)) if n >= 2 else 0.0
        return mu / (sigma + self.eps)


class SortinoPerf:
    """
    Универсальный Sortino с переключателем «режима задачи».

    mode = "good":
        perf_value = mean(R) / (downside_dev(R) + eps),
        downside_dev^2 = mean( r_i^2 | r_i < 0 ), классический Sortino.
        Подходит для поиска ХОРОШИХ режимов: штрафуем «плохой хвост».

    mode = "bad":
        perf_value = mean(R) / (upside_dev(R) + eps),
        upside_dev^2 = mean( r_i^2 | r_i > 0 ).
        Зеркальный вариант: при поиске ПЛОХИХ режимов штрафуем «хороший хвост»,
        чтобы метрика сильнее выделяла области с «плохим профилем».
    """
    def __init__(self, mode: str = "good", eps: float = 1e-12):
        assert mode in ("good", "bad"), "SortinoPerf.mode must be 'good' or 'bad'"
        self.mode = mode
        self.eps = eps

    def __call__(self, R: np.ndarray, T: Optional[np.ndarray] = None) -> float:
        R = np.asarray(R, dtype=float)
        n = R.size
        if n == 0:
            return 0.0

        mu = float(np.mean(R))

        if self.mode == "good":
            tail = R[R < 0.0]  # downside deviation
        else:  # self.mode == "bad"
            tail = R[R > 0.0]  # upside deviation (зеркальный Sortino)

        if tail.size == 0:
            dev = 0.0
        else:
            # root(mean(tail^2))
            dev = float(np.sqrt(np.mean(tail * tail)))

        return mu / (dev + self.eps)


class EdgePerf:
    """
    perf_value = p*G - (1-p)*L
      p = mean(R > 0)
      G = mean(R | R>0)
      L = mean(|R| | R<0)
    Удобно для асимметричных профилей выигрыша/проигрыша.
    """
    def __init__(self, eps: float = 1e-12):
        self.eps = eps  # оставлен на случай расширений; здесь напрямую не используется

    def __call__(self, R: np.ndarray, T: Optional[np.ndarray] = None) -> float:
        R = np.asarray(R, dtype=float)
        n = R.size
        if n == 0:
            return 0.0
        pos = R[R > 0.0]
        neg = R[R < 0.0]
        p = float(pos.size) / float(n)
        G = float(np.mean(pos)) if pos.size else 0.0
        L = float(np.mean(np.abs(neg))) if neg.size else 0.0
        return p * G - (1.0 - p) * L


class KellyProxyPerf:
    """
    perf_value ≈ mean(R) - 0.5 * var(R), when |R| is small.
    Приближённый лог-рост (удобно в некоторых задачах).
    """
    def __init__(self):
        pass

    def __call__(self, R: np.ndarray, T: Optional[np.ndarray] = None) -> float:
        R = np.asarray(R, dtype=float)
        n = R.size
        if n == 0:
            return 0.0
        mu = float(np.mean(R))
        var = float(np.var(R, ddof=1)) if n >= 2 else 0.0
        return mu - 0.5 * var


# ========= Stability metric (Stab–BIN × Stab–COV) =========

class UVxCoverageStability:
    """
    Возвращает словарь:
      stab_bin = exp( -gamma * V / (f_hat*(1-f_hat) + eps) )
      stab_cov = (tau + covered_bins) / (B + 2*tau)
      stability_value = stab_bin * stab_cov

    Где:
      - Бины заданы последовательным equal-count разбиением через размерности s (len(s)=B).
      - f_j = mean(R_j < 0) по НЕпустым бинам; w_j ∝ size_j; V = sum_j w_j (f_j - f_hat)^2.
      - covered_bins = # { j : size_j >= m_min } — считаются по ВСЕМ B бинам (включая нулевые).

    Интерфейс совместим с деревом:
      reliab_metric(R, T, B, s) -> dict(...)
    """
    def __init__(self, gamma: float = 1.0, tau: float = 1.0, m_min: int = 1, eps: float = 1e-12):
        self.gamma = gamma
        self.tau = tau
        self.m_min = m_min
        self.eps = eps

    def __call__(self, R: np.ndarray, T: np.ndarray, B: int, s: np.ndarray) -> Dict[str, Any]:
        R = np.asarray(R, dtype=float)
        s = np.asarray(s, dtype=int)
        if s.size != B:
            raise ValueError("s must have length B")

        # --- coverage по всем B бинам (включая нулевые) ---
        covered = int(np.sum(s >= self.m_min))
        stab_cov = (self.tau + covered) / (B + 2.0 * self.tau)

        # --- U(V) только по НЕпустым бинам ---
        if R.size == 0:
            # данных нет -> U(V)=1.0; присутствие отразится в coverage
            stab_bin = 1.0
            return {
                "stab_bin": float(stab_bin),
                "stab_cov": float(stab_cov),
                "stability_value": float(stab_bin * stab_cov),
            }

        # Индексы начала/конца бинов
        starts = np.cumsum(np.r_[0, s[:-1]])
        ends   = starts + s
        nonempty_mask = s > 0
        sizes = s[nonempty_mask]
        if sizes.size == 0:
            stab_bin = 1.0
            return {
                "stab_bin": float(stab_bin),
                "stab_cov": float(stab_cov),
                "stability_value": float(stab_bin * stab_cov),
            }

        # f_j = mean(R_bin < 0)
        f_list = []
        for st, en, ok in zip(starts, ends, nonempty_mask):
            if not ok:
                continue
            Rij = R[st:en]
            if Rij.size == 0:
                continue
            f_list.append(float(np.mean(Rij < 0.0)))

        if len(f_list) == 0:
            stab_bin = 1.0
            return {
                "stab_bin": float(stab_bin),
                "stab_cov": float(stab_cov),
                "stability_value": float(stab_bin * stab_cov),
            }

        f_arr = np.asarray(f_list, dtype=float)
        w = sizes.astype(float) / float(np.sum(sizes))
        f_hat = float(np.sum(w * f_arr))
        V = float(np.sum(w * (f_arr - f_hat) ** 2))
        denom = f_hat * (1.0 - f_hat) + self.eps
        stab_bin = float(np.exp(-self.gamma * (V / denom)))

        stability_value = stab_bin * stab_cov
        return {
            "stab_bin": float(stab_bin),
            "stab_cov": float(stab_cov),
            "stability_value": float(stability_value),
            # опционально диагностика:
            # "f_hat": f_hat, "V": V, "bins_nonempty": int(sizes.size)
        }
