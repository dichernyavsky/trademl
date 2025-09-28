"""
Простой пример для копирования в Jupyter notebook.
Добавьте этот код после вашего существующего кода.
"""

# Импортируем функцию для расчета весов
from ml.sample_weights import get_sample_weights_for_ml

# Ваш существующий код остается без изменений
trades_df = trades["BTCUSDT"]

trades_long = trades_df[trades_df.direction==1].dropna(subset = feats)
trades_short = trades_df[trades_df.direction==-1].dropna(subset = feats)

trades_long["target"] = trades_long["bin"].replace(-1, 0)
trades_short["target"] = trades_short["bin"].replace(-1, 0).astype(int)

# Данные
trades_ml = trades_long
X = trades_ml[feats]
y = trades_ml['target']

# === ДОБАВЛЯЕМ РАСЧЕТ SAMPLE WEIGHTS ===
print("Calculating sample weights...")

# Предполагаем, что у вас есть close_prices для BTCUSDT
# Если нет, создайте их из ваших данных
# close_prices = your_close_prices_series  # Series с ценами закрытия

# Рассчитываем веса
sample_weights = get_sample_weights_for_ml(
    events=trades_ml,           # DataFrame с трейдами
    close=close_prices,         # Series с ценами (адаптируйте под ваши данные)
    weight_type='w_event',      # тип весов
    t1_col='exit_time'          # колонка с временем выхода (адаптируйте под ваши данные)
)

print(f"Sample weights calculated: min={sample_weights.min():.4f}, max={sample_weights.max():.4f}, mean={sample_weights.mean():.4f}")

# Инициализация модели (без изменений)
clf = XGBClassifier(
    n_estimators=500,         
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    objective="binary:logistic",  
    eval_metric="logloss"
)

cv = KFold(n_splits=4, shuffle=False)

# Подготовим массивы для всех предсказаний
y_true_all = np.zeros_like(y)
y_pred_all = np.zeros_like(y)
all_val_idx = []

# Цикл по фолдам (ЕДИНСТВЕННОЕ ИЗМЕНЕНИЕ - добавляем sample_weight)
for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # === ИЗМЕНЕНИЕ: добавляем sample_weight ===
    train_weights = sample_weights[train_idx]
    clf.fit(X_train, y_train, sample_weight=train_weights)  # Добавили sample_weight
    # === КОНЕЦ ИЗМЕНЕНИЯ ===
    
    y_pred = clf.predict(X_val)

    y_true_all[val_idx] = y_val
    y_pred_all[val_idx] = y_pred
    all_val_idx.extend(val_idx)

    print(f"Fold {fold+1} — Accuracy: {(y_val == y_pred).mean():.3f}")

# Финальный отчёт
print("\n=== Classification Report ===")
print(classification_report(y.iloc[all_val_idx], y_pred_all[all_val_idx]))

# Добавим в исходную таблицу трейдов
trades_ml["y_true"] = y
trades_ml["y_pred"] = y_pred_all
trades_ml["sample_weights"] = sample_weights  # Добавили веса в DataFrame

print("\nSample weights added to trades_ml DataFrame!") 