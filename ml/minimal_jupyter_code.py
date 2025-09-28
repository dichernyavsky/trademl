# === МИНИМАЛЬНЫЙ КОД ДЛЯ ДОБАВЛЕНИЯ SAMPLE WEIGHTS ===

# 1. Импорт
from ml.sample_weights import get_sample_weights_for_ml

# 2. Расчет весов (добавить после подготовки данных)
sample_weights = get_sample_weights_for_ml(
    events=trades_ml,           # ваш DataFrame с трейдами
    close=close_prices,         # Series с ценами закрытия
    weight_type='w_event',      # тип весов
    t1_col='exit_time'          # колонка с временем выхода
)

# 3. Изменить в цикле cross-validation (единственное изменение)
for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    # ДОБАВИТЬ ЭТУ СТРОКУ:
    train_weights = sample_weights[train_idx]
    
    # ИЗМЕНИТЬ ЭТУ СТРОКУ:
    clf.fit(X_train, y_train, sample_weight=train_weights)  # добавили sample_weight
    
    y_pred = clf.predict(X_val)
    # ... остальной код без изменений

# 4. Добавить веса в DataFrame (опционально)
trades_ml["sample_weights"] = sample_weights

# === ГОТОВО! === 