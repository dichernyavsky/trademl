сделаем «важность фичей» как сумму фактического выигрыша сплита по этой фиче, накопленного по всему дереву (опц. с весом по глубине), и вернём в удобном pd.Series. Это естественная метрика для твоего дерева, потому что именно gain — то, что оптимизируется при выборе сплита. Ниже — минимальные точечные правки.

Что добавить в tree.py

В __init__ — хранилища:

self._feat_gain = {}            # суммарный gain по фиче
self._feat_gain_depth = {}      # суммарный gain с весом по глубине
self._feat_splits = {}          # счётчик сплитов по фиче


В начале fit(...) — обнулить:

self._feat_gain.clear(); self._feat_gain_depth.clear(); self._feat_splits.clear()


В _build_recursive(...) — сразу после того как выбрали валидный best и ДО рекурсии в детей, аккумулируем важность:

# internal node
self.nodes_.append(Node(depth, node_idx, idx, split, None, None, False, list(parent_rule), st))

# === важность фичей ===
g = float(gain)
f = split.feature
w_depth = 1.0 / (1.0 + depth)      # мягкий вес за глубину (root=1, глубже — меньше)

self._feat_gain[f] = self._feat_gain.get(f, 0.0) + g
self._feat_gain_depth[f] = self._feat_gain_depth.get(f, 0.0) + w_depth * g
self._feat_splits[f] = self._feat_splits.get(f, 0) + 1


Добавить публичный метод в класс:

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


Эти вставки не меняют публичный API и не затрагивают критерий — только собирают статистику во время обучения дерева (в местах показанных выше) 

tree

.

Как пользоваться
imps = tree.feature_importances(normalize=True, depth_weighted=False)   # сумма gain, нормирована
imps_dw = tree.feature_importances(normalize=True, depth_weighted=True) # с весом по глубине
counts = tree.feature_split_counts()

print(imps)     # важности (доля вклада в общий gain)
print(imps_dw)  # то же, но root-сплиты ценятся сильнее
print(counts)   # просто сколько раз фича сплитилась

