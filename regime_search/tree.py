# BayesBadTree: decision tree that finds subgroups minimizing or maximizing external score Q
# (product of performance and reliability metrics)
# Dependencies: numpy, pandas
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import math

# -------------------- data structures --------------------
@dataclass
class Split:
    feature: str
    kind: str  # 'numeric' or 'categorical'
    threshold: Optional[float] = None
    cat_subset: Optional[frozenset] = None
    op: Optional[str] = None  # NEW: '<=', '>', 'in', 'not in'

    def describe(self) -> str:
        if self.kind == "numeric":
            if self.op in ("<=", ">"):
                return f"{self.feature} {self.op} {self.threshold:.6g}"
            # fallback (на всякий)
            return f"{self.feature} <= {self.threshold:.6g}"
        # categorical
        cats = sorted(list(self.cat_subset)) if self.cat_subset else []
        if self.op == "not in":
            return f"{self.feature} not in {cats}"
        return f"{self.feature} in {cats}"

@dataclass
class NodeStats:
    n: int
    mean_ret: float
    std_ret: float
    sharpe: float

@dataclass
class Node:
    depth: int
    idx: int  # node index in nodes_ list
    sample_idx: np.ndarray  # indices of samples in this node
    split: Optional[Split]
    left: Optional[int]
    right: Optional[int]
    leaf: bool
    rule_pos_path: List[Split]   # only positive-path clauses for readability
                                      # (left children get split conditions, right children inherit parent rules)
    stats: NodeStats

# -------------------- main class --------------------
class Tree:
    """
    Decision tree that finds subgroups minimizing or maximizing external leaf_score
    (product of performance metric (Sharpe/Sortino) and stability metric (Stab–BIN × Stab–COV)).

    Uses external performance and stability metrics to evaluate splits.
    Supports both "bad" (HardMin) and "good" (HardMax) modes.

    Parameters
    ----------
    feature_info : dict
        Mapping: feature -> {'type': 'numeric' | 'categorical'}.
    max_depth : int
        Maximum tree depth (root = 0).
    min_samples_leaf : int
        Minimal number of samples required in each child leaf for a split to be valid.
    max_thresholds : int
        Max number of candidate thresholds per numeric feature (quantile-based).
    root_feature : Optional[str]
        If provided, force the root split to use this feature.
    random_state : int
        For any randomized tie-breakers (currently not used, but kept for reproducibility).
    perf_metric : callable
        Performance metric callback: U = perf_metric(R, T) where R=returns, T=time indices.
    reliab_metric : callable
        Reliability metric callback: G_info = reliab_metric(R, T, B, s) where B=n_bins, s=bin_sizes.
    mode : str
        "bad" for HardMin (find worst states), "good" for HardMax (find best states).
    n_bins : int
        Target number of equal-count bins for sequential binning.
    delta_min : float
        Optional threshold for Q gap filtering.
    w_target_min : float
        Optional threshold for target child weight filtering (min for bad, max for good).
    sample_ids : Optional[np.ndarray]
        Original sample identifiers (will be set in fit() method).
    """

    def __init__(self,
                 feature_info: Dict[str, Dict[str, Any]],
                 max_depth: int = 3,
                 min_samples_leaf: int = 200,
                 max_thresholds: int = 32,
                 root_feature: Optional[str] = None,
                 random_state: int = 0,
                 # --- metric interfaces ---
                 perf_metric=None,      # performance metric callback
                 reliab_metric=None,   # reliability metric callback  
                 mode: str = "bad",    # "bad" for HardMin, "good" for HardMax
                 # --- binning and filtering ---
                 n_bins: int = 40,     # target number of equal-count bins
                 delta_min: float = 0.0,
                 w_target_min: float = 0.0,
                 sample_ids: Optional[np.ndarray] = None):
        # Convert new format to old format if needed
        if "features_numeric" in feature_info or "features_categorical" in feature_info:
            numeric = feature_info.get("features_numeric", [])
            categorical = feature_info.get("features_categorical", [])
            feature_info = {f: {"type": "numeric"} for f in numeric}
            feature_info.update({f: {"type": "categorical"} for f in categorical})
        self.feature_info = feature_info
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_thresholds = int(max_thresholds)
        self.root_feature = root_feature
        self.random_state = int(random_state)

        # Metric interfaces
        if perf_metric is None:
            raise ValueError("perf_metric must be provided")
        if reliab_metric is None:
            raise ValueError("reliab_metric must be provided")
        self.perf_metric = perf_metric
        self.reliab_metric = reliab_metric
        self.mode = mode

        # Binning and filtering
        self.n_bins = int(n_bins)
        self.delta_min = float(delta_min)
        self.w_target_min = float(w_target_min)
        self.sample_ids = sample_ids

        self.nodes_: List[Node] = []
        self._fitted = False
        
        # Feature importance tracking
        self._feat_gain = {}            # суммарный gain по фиче
        self._feat_gain_depth = {}      # суммарный gain с весом по глубине
        self._feat_splits = {}          # счётчик сплитов по фиче

    # -------------------- public API --------------------
    def fit(self, X: pd.DataFrame, returns: Union[pd.Series, np.ndarray], 
            sample_ids: np.ndarray):
        """
        Fit the decision tree to identify market states minimizing (mode='bad') or 
        maximizing (mode='good') the external leaf score (performance × stability).
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        returns : Union[pd.Series, np.ndarray]
            Return values for each sample.
        sample_ids : np.ndarray
            Original sample identifiers (e.g., trade IDs or DataFrame index).
            Must be provided to preserve true IDs for tracking.
            
        Returns
        -------
        self : BayesBadTreeSharpe
            Fitted tree instance.
        """
        X = X.copy()
        y_r = np.asarray(returns, dtype=float)
        assert X.shape[0] == y_r.shape[0], "X and returns must align"

        n = X.shape[0]
        # Handle sample_ids - now required to preserve true IDs
        if sample_ids is None:
            raise ValueError("sample_ids must be provided to preserve true IDs.")
        
        sid = np.asarray(sample_ids)
        assert sid.shape[0] == n, "sample_ids must match returns length"
        self.sample_ids = sid
        self._y_r = y_r  # сохраняем исходный вектор доходностей для последующего расчёта стабильности в листах
        
        # Clear feature importance tracking
        self._feat_gain.clear()
        self._feat_gain_depth.clear()
        self._feat_splits.clear()
        
        # Root node contains all samples (positions 0..N-1)
        idx_all = np.arange(n, dtype=int)
        self.nodes_.clear()
        self._build_recursive(X, y_r, idx_all, depth=0, parent_rule=[], force_feature=self.root_feature)
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.Series:
        """Return leaf (state) id for each row; -1 if not fitted."""
        if not self._fitted:
            return pd.Series(-1, index=X.index, name="state_id")
        leaf_ids = np.fromiter((self._traverse_row(X.iloc[i]) for i in range(X.shape[0])),
                               dtype=int, count=X.shape[0])
        return pd.Series(leaf_ids, index=X.index, name="state_id")

    def leaf_report(self) -> pd.DataFrame:
        """Tabulate leaves with rules and metrics using external metric callbacks."""
        rows = []
        for node in self.nodes_:
            if node.leaf:
                st = node.stats
                original_ids = self.sample_ids[node.sample_idx]

                # Prepare data for metrics
                R_leaf = self._y_r[node.sample_idx]
                T_leaf = np.arange(len(R_leaf))  # T — просто монотонная последовательность индексов; реально используется только порядок для последовательного бинирования
                B = self.n_bins
                s_leaf = self._create_equal_count_bins(len(R_leaf), B)
                
                # Calculate metrics using external callbacks
                perf_leaf = self.perf_metric(R_leaf, T_leaf)
                stab_info_leaf = self.reliab_metric(R_leaf, T_leaf, B, s_leaf)
                stability_leaf = stab_info_leaf.get("stability_value", 
                    stab_info_leaf.get("stab_bin", 1.0) * stab_info_leaf.get("stab_cov", 1.0))
                leaf_score_leaf = perf_leaf * stability_leaf

                row = {
                    "leaf_id": node.idx,
                    "depth": node.depth,
                    "rule": self._rule_to_string(node.rule_pos_path),
                    "n": st.n,
                    "mean_ret": st.mean_ret,
                    "std_ret": st.std_ret,
                    "sharpe": st.sharpe,     # μ/σ (no √n) - for backward compatibility
                    # --- external metric fields ---
                    "perf_value": perf_leaf,
                    "stab_bin": stab_info_leaf.get("stab_bin", 1.0),
                    "stab_cov": stab_info_leaf.get("stab_cov", 1.0),
                    "stability_value": stability_leaf,
                    "leaf_score": leaf_score_leaf,
                    # --- configuration fields ---
                    "mode": self.mode,
                    "n_bins": self.n_bins,
                }
                row["indices"] = original_ids
                rows.append(row)

        # Sort based on mode
        if self.mode == "bad":
            df = pd.DataFrame(rows).sort_values(["leaf_score", "n"], ascending=[True, False]).reset_index(drop=True)
        else:  # mode == "good"
            df = pd.DataFrame(rows).sort_values(["leaf_score", "n"], ascending=[False, False]).reset_index(drop=True)
        return df

    def export_rules_text(self) -> str:
        """Human-readable rules sorted by leaf_score (asc for bad, desc for good)."""
        rep = self.leaf_report()
        lines = []
        for _, row in rep.iterrows():
            lines.append(
                f"[leaf {int(row.leaf_id)} | depth {int(row.depth)} | n={int(row.n)}] "
                f"Score={row.leaf_score:.4g} | Perf={row.perf_value:.4g} | "
                f"Stab={row.stability_value:.4g} (Stab–BIN={row.stab_bin:.4g}, Stab–COV={row.stab_cov:.4g}) :: {row.rule}"
            )
        return "\n".join(lines)

    def export_rules_json(self) -> List[Dict[str, Any]]:
        rep = self.leaf_report()
        out = []
        for _, row in rep.iterrows():
            leaf_id = int(row.leaf_id)
            out.append({
                "leaf_id": leaf_id,
                "depth": int(row.depth),
                "n": int(row.n),
                "perf_value": float(row.perf_value),
                "stab_bin": float(row.stab_bin),
                "stab_cov": float(row.stab_cov),
                "stability_value": float(row.stability_value),
                "leaf_score": float(row.leaf_score),
                "rule_text": str(row.rule),
                "rule_clauses": [self._split_to_dict(sp) for sp in self.nodes_[leaf_id].rule_pos_path],
            })
        return out

    def predict_leaf_ids(self, X: pd.DataFrame) -> np.ndarray:
        """Return array of leaf_id for each sample (by position in original array)."""
        if not self._fitted:
            return np.full(X.shape[0], -1, dtype=np.int32)
        
        leaf_ids = np.empty(X.shape[0], dtype=np.int32)
        
        def assign_leaf_ids(node_idx: int):
            node = self.nodes_[node_idx]
            if node.leaf:
                # Assign leaf_id to all samples in this node
                leaf_ids[node.sample_idx] = node.idx
            else:
                # Recursively assign to children
                if node.left is not None:
                    assign_leaf_ids(node.left)
                if node.right is not None:
                    assign_leaf_ids(node.right)
        
        assign_leaf_ids(0)  # Start from root
        return leaf_ids

    def leaf_index_map(self) -> Dict[int, np.ndarray]:
        """Return mapping from leaf_id to original sample indices."""
        mapping = {}
        
        def collect_leaf_indices(node_idx: int):
            node = self.nodes_[node_idx]
            if node.leaf:
                # Convert sample positions to original sample_ids
                mapping[node.idx] = self.sample_ids[node.sample_idx]
            else:
                # Recursively collect from children
                if node.left is not None:
                    collect_leaf_indices(node.left)
                if node.right is not None:
                    collect_leaf_indices(node.right)
        
        collect_leaf_indices(0)  # Start from root
        return mapping

    def export_leaf_membership(self, trades_df: pd.DataFrame, 
                              feature_cols: List[str], 
                              id_col: Optional[str] = None) -> pd.DataFrame:
        """Add leaf_id column to trades DataFrame."""
        # Extract features for prediction
        X = trades_df[feature_cols]
        # Use real tree traversal for out-of-sample data
        leaf_ids = self.transform(X).to_numpy()
        
        # Create output DataFrame
        out = trades_df.copy()
        out["leaf_id"] = leaf_ids
        
        return out

    # -------------------- core recursion --------------------
    def _build_recursive(self, X: pd.DataFrame, y_r: np.ndarray,
                         idx: np.ndarray, depth: int, parent_rule: List[Split],
                         force_feature: Optional[str]) -> int:
        node_idx = len(self.nodes_)
        st = self._node_stats(y_r[idx])

        # stop conditions
        if (depth >= self.max_depth) or (idx.size < 2 * self.min_samples_leaf):
            self.nodes_.append(Node(depth, node_idx, idx, None, None, None, True, list(parent_rule), st))
            return node_idx

        # find best split
        best = self._find_best_split(X, y_r, idx, st, force_feature)
        if best is None:
            self.nodes_.append(Node(depth, node_idx, idx, None, None, None, True, list(parent_rule), st))
            return node_idx

        split, left_idx, right_idx, gain = best
        if (left_idx.size < self.min_samples_leaf) or (right_idx.size < self.min_samples_leaf) or (gain <= 0.0):
            self.nodes_.append(Node(depth, node_idx, idx, None, None, None, True, list(parent_rule), st))
            return node_idx

        # internal node
        self.nodes_.append(Node(depth, node_idx, idx, split, None, None, False, list(parent_rule), st))
        
        # === важность фичей ===
        g = float(gain)
        f = split.feature
        w_depth = 1.0 / (1.0 + depth)      # мягкий вес за глубину (root=1, глубже — меньше)

        self._feat_gain[f] = self._feat_gain.get(f, 0.0) + g
        self._feat_gain_depth[f] = self._feat_gain_depth.get(f, 0.0) + w_depth * g
        self._feat_splits[f] = self._feat_splits.get(f, 0) + 1
        
        # LEFT child gets positive clause
        left_clause = Split(split.feature, split.kind, split.threshold, split.cat_subset,
                            "<=" if split.kind=="numeric" else "in")
        left_rule = list(parent_rule) + [left_clause]

        # RIGHT child gets negative clause
        right_clause = Split(split.feature, split.kind, split.threshold, split.cat_subset,
                             ">" if split.kind=="numeric" else "not in")
        right_rule = list(parent_rule) + [right_clause]

        left_child = self._build_recursive(X, y_r, left_idx, depth + 1, left_rule, None)
        right_child = self._build_recursive(X, y_r, right_idx, depth + 1, right_rule, None)
        self.nodes_[node_idx].left = left_child
        self.nodes_[node_idx].right = right_child
        return node_idx

    def _find_best_split(self, X: pd.DataFrame, y_r: np.ndarray,
                         idx: np.ndarray, parent_stats: NodeStats,
                         force_feature: Optional[str]):
        best_gain = -np.inf
        best_tuple = None
        nP = len(idx)
        
        # Cache parent metrics (calculated once for all candidates)
        R_P = y_r[idx]
        T_P = np.arange(len(idx))  # T — просто монотонная последовательность индексов; реально используется только порядок для последовательного бинирования
        B = self.n_bins
        s_P = self._create_equal_count_bins(len(idx), B)
        perf_P = self.perf_metric(R_P, T_P)
        stab_info_P = self.reliab_metric(R_P, T_P, B, s_P)
        stability_P = stab_info_P.get("stability_value", 
            stab_info_P.get("stab_bin", 1.0) * stab_info_P.get("stab_cov", 1.0))
        leaf_score_P = perf_P * stability_P

        features = [force_feature] if force_feature is not None else list(self.feature_info.keys())
        for feat in features:
            kind = self.feature_info[feat]["type"]
            col = X.iloc[idx][feat]

            if kind == "numeric":
                for thr in self._numeric_thresholds(col, self.max_thresholds):
                    left_mask = col <= thr
                    right_mask = ~left_mask
                    L = idx[left_mask.values]
                    R = idx[right_mask.values]
                    if (L.size < self.min_samples_leaf) or (R.size < self.min_samples_leaf):
                        continue
                    # Prepare data for metrics
                    r_L, r_R = y_r[L], y_r[R]
                    T_L = np.arange(len(L))  # T — просто монотонная последовательность индексов; реально используется только порядок для последовательного бинирования
                    T_R = np.arange(len(R))  # T — просто монотонная последовательность индексов; реально используется только порядок для последовательного бинирования
                    B = self.n_bins
                    s_L = self._create_equal_count_bins(len(L), B)
                    s_R = self._create_equal_count_bins(len(R), B)
                    
                    # Calculate metrics using external callbacks
                    perf_L = self.perf_metric(r_L, T_L)
                    perf_R = self.perf_metric(r_R, T_R)
                    
                    stab_info_L = self.reliab_metric(r_L, T_L, B, s_L)
                    stab_info_R = self.reliab_metric(r_R, T_R, B, s_R)
                    
                    # Calculate stability and leaf scores
                    stability_L = stab_info_L.get("stability_value", 
                        stab_info_L.get("stab_bin", 1.0) * stab_info_L.get("stab_cov", 1.0))
                    stability_R = stab_info_R.get("stability_value", 
                        stab_info_R.get("stab_bin", 1.0) * stab_info_R.get("stab_cov", 1.0))
                    leaf_score_L = perf_L * stability_L
                    leaf_score_R = perf_R * stability_R
                    
                    # Calculate gain based on mode (using cached leaf_score_P)
                    wL = L.size / nP; wR = 1.0 - wL
                    if self.mode == "bad":  # HardMin
                        if leaf_score_L <= leaf_score_R:
                            target_score, w_target = leaf_score_L, wL
                        else:
                            target_score, w_target = leaf_score_R, wR
                        gain = leaf_score_P - target_score
                    else:  # "good", HardMax
                        if leaf_score_L >= leaf_score_R:
                            target_score, w_target = leaf_score_L, wL
                        else:
                            target_score, w_target = leaf_score_R, wR
                        gain = target_score - leaf_score_P
                    
                    # Optional protections
                    if self.delta_min > 0.0 and gain < self.delta_min:
                        continue
                    if self.w_target_min > 0.0 and w_target < self.w_target_min:
                        continue
                    if gain > best_gain:
                        best_gain = gain
                        best_tuple = (Split(feat, "numeric", float(thr), None), L, R, gain)

            else:  # categorical: greedy growing subset to maximize gain
                categories = pd.Categorical(col).categories.tolist()
                if len(categories) <= 1:
                    continue
                remaining = set(categories)
                subset: set = set()
                improved = True
                best_local_gain = -np.inf
                best_local_subset = None
                best_local_lr = None
                while improved and remaining:
                    improved = False
                    best_cat = None
                    for c in list(remaining):
                        trial = subset | {c}
                        left_mask = col.isin(trial)
                        right_mask = ~left_mask
                        L = idx[left_mask.values]
                        R = idx[right_mask.values]
                        if (L.size < self.min_samples_leaf) or (R.size < self.min_samples_leaf):
                            continue
                        # Prepare data for metrics
                        r_L, r_R = y_r[L], y_r[R] # returns for left and right child nodes
                        T_L = np.arange(len(L))  # T — просто монотонная последовательность индексов; реально используется только порядок для последовательного бинирования
                        T_R = np.arange(len(R))  # T — просто монотонная последовательность индексов; реально используется только порядок для последовательного бинирования
                        B = self.n_bins
                        s_L = self._create_equal_count_bins(len(L), B)
                        s_R = self._create_equal_count_bins(len(R), B)
                        
                        # Calculate metrics using external callbacks
                        perf_L = self.perf_metric(r_L, T_L)
                        perf_R = self.perf_metric(r_R, T_R)
                        
                        stab_info_L = self.reliab_metric(r_L, T_L, B, s_L)
                        stab_info_R = self.reliab_metric(r_R, T_R, B, s_R)
                        
                        # Calculate stability and leaf scores
                        stability_L = stab_info_L.get("stability_value", 
                            stab_info_L.get("stab_bin", 1.0) * stab_info_L.get("stab_cov", 1.0))
                        stability_R = stab_info_R.get("stability_value", 
                            stab_info_R.get("stab_bin", 1.0) * stab_info_R.get("stab_cov", 1.0))
                        leaf_score_L = perf_L * stability_L
                        leaf_score_R = perf_R * stability_R
                        
                        # Calculate gain based on mode (using cached leaf_score_P)
                        wL = L.size / nP; wR = 1.0 - wL
                        if self.mode == "bad":  # HardMin
                            if leaf_score_L <= leaf_score_R:
                                target_score, w_target = leaf_score_L, wL
                            else:
                                target_score, w_target = leaf_score_R, wR
                            gain = leaf_score_P - target_score
                        else:  # "good", HardMax
                            if leaf_score_L >= leaf_score_R:
                                target_score, w_target = leaf_score_L, wL
                            else:
                                target_score, w_target = leaf_score_R, wR
                            gain = target_score - leaf_score_P
                        
                        # Same optional filters
                        if self.delta_min > 0.0 and gain < self.delta_min:
                            continue
                        if self.w_target_min > 0.0 and w_target < self.w_target_min:
                            continue
                        if gain > best_local_gain + 1e-12:
                            best_local_gain = gain
                            best_cat = c
                            best_local_subset = set(trial)
                            best_local_lr = (L, R)
                    if best_cat is not None:
                        subset.add(best_cat)
                        remaining.remove(best_cat)
                        improved = True
                if (best_local_subset is not None) and (best_local_gain > best_gain):
                    best_gain = best_local_gain
                    best_tuple = (
                        Split(feat, "categorical", None, frozenset(best_local_subset)),
                        best_local_lr[0], best_local_lr[1], best_local_gain
                    )
        return best_tuple

    # -------------------- helpers --------------------
    def _create_equal_count_bins(self, n: int, B: int) -> np.ndarray:
        """
        Create equal-count bin sizes for sequential binning.
        
        Parameters
        ----------
        n : int
            Total number of samples
        B : int
            Target number of bins
            
        Returns
        -------
        np.ndarray
            Array of bin sizes (may have trailing zeros if n < B)
        """
        if n <= 0:
            return np.zeros(B, dtype=int)
        
        q = n // B  # base size per bin
        r = n - B * q  # remainder
        
        # First r bins get q+1 samples, remaining bins get q samples
        sizes = np.full(B, q, dtype=int)
        if r > 0:
            sizes[:r] = q + 1
            
        return sizes


    @staticmethod
    def _finite(x: float) -> float:
        """Convert infinite/NaN values to 0.0 for numerical stability."""
        return x if np.isfinite(x) else 0.0

    # -------------------- stats & helpers --------------------
    def _node_stats(self, rets: np.ndarray) -> NodeStats:
        n = int(rets.size)
        eps = 1e-12
        if n == 0:
            m = float("nan"); s = float("nan"); sh = float("nan")
        elif n == 1:
            m = float(rets[0]); s = 0.0; sh = m / (s + eps)
        else:
            m = float(np.mean(rets))
            s = float(np.std(rets, ddof=1))
            sh = m / (s + eps)
        return NodeStats(n=n, mean_ret=m, std_ret=s, sharpe=sh)

    @staticmethod
    def _numeric_thresholds(col: pd.Series, max_thresholds: int) -> List[float]:
        """
        Generate candidate thresholds for numeric features.
        
        Uses 5%-95% quantile range to avoid extreme splits that would create
        very small or very large child nodes.
        """
        vals = pd.to_numeric(col, errors="coerce").to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return []
        # evenly spaced quantiles in (5%, 95%] to avoid extreme tiny splits
        qs = np.linspace(0.05, 0.95, num=min(max_thresholds, 19))
        thr = np.unique(np.quantile(vals, qs))
        thr = thr[~np.isnan(thr)]
        return [float(t) for t in np.unique(thr)]

    def _traverse_row(self, row: pd.Series) -> int:
        node_idx = 0
        while True:
            node = self.nodes_[node_idx]
            if node.leaf or node.split is None:
                return node.idx
            sp = node.split
            go_left = (row[sp.feature] <= sp.threshold) if sp.kind == "numeric" else (row[sp.feature] in sp.cat_subset)
            node_idx = node.left if go_left else node.right

    @staticmethod
    def _rule_to_string(rule: List[Split]) -> str:
        """
        Convert rule path (now with both positive and negative clauses) to a readable string.
        """
        if not rule:
            return "<ALL>"
        return " AND ".join(sp.describe() for sp in rule)

    @staticmethod
    def _split_to_dict(sp: Split) -> Dict[str, Any]:
        if sp.kind == "numeric":
            return {"feature": sp.feature, "op": sp.op or "<=", "threshold": float(sp.threshold)}
        return {"feature": sp.feature, "op": sp.op or "in", "values": sorted(list(sp.cat_subset))}

    def feature_importances(self, normalize: bool = True, depth_weighted: bool = False) -> pd.Series:
        """
        Возвращает важности фичей как сумму gain по сплитам этой фичи.
        depth_weighted=True берёт сумму gain с весом 1/(1+depth).
        normalize=True нормирует на сумму = 1.0.
        """
        if not self._fitted:
            return pd.Series(dtype=float)
        src = self._feat_gain_depth if depth_weighted else self._feat_gain
        if len(src) == 0:
            return pd.Series(dtype=float)
        s = pd.Series(src, dtype=float).sort_values(ascending=False)
        if normalize:
            total = s.sum()
            if total > 0:
                s = s / total
        return s

    def feature_split_counts(self) -> pd.Series:
        """Сколько раз каждая фича использовалась в сплитах."""
        if not self._fitted:
            return pd.Series(dtype=int)
        return pd.Series(self._feat_splits, dtype=int).sort_values(ascending=False)
