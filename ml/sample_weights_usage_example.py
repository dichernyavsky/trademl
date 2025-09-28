"""
Пример использования sample weights с существующим ML пайплайном.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sample_weights import get_sample_weights_for_ml


def train_with_sample_weights(trades_df, feats, close_prices):
    """
    Обучение модели с sample weights.
    
    Args:
        trades_df: DataFrame с трейдами
        feats: список фич
        close_prices: Series с ценами закрытия
    """
    
    # Подготовка данных (как в вашем коде)
    trades_long = trades_df[trades_df.direction == 1].dropna(subset=feats)
    trades_short = trades_df[trades_df.direction == -1].dropna(subset=feats)
    
    trades_long["target"] = trades_long["bin"].replace(-1, 0)
    trades_short["target"] = trades_short["bin"].replace(-1, 0).astype(int)
    
    # Используем long trades (как в вашем примере)
    trades_ml = trades_long
    X = trades_ml[feats]
    y = trades_ml['target']
    
    # Рассчитываем sample weights
    print("Calculating sample weights...")
    sample_weights = get_sample_weights_for_ml(
        events=trades_ml,  # DataFrame с трейдами
        close=close_prices,  # Series с ценами
        weight_type='w_event',  # тип весов
        t1_col='exit_time'  # колонка с временем выхода (адаптируйте под ваши данные)
    )
    print(f"Sample weights: min={sample_weights.min():.4f}, max={sample_weights.max():.4f}, mean={sample_weights.mean():.4f}")
    
    # Инициализация модели
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
    
    # Индексы, чтобы потом построить DataFrame
    all_val_idx = []
    
    # Цикл по фолдам
    for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # Получаем веса для тренировочных данных
        train_weights = sample_weights[train_idx]
        
        # Обучаем модель с весами
        clf.fit(X_train, y_train, sample_weight=train_weights)
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
    trades_ml["sample_weights"] = sample_weights
    
    return trades_ml, clf


def compare_with_and_without_weights(trades_df, feats, close_prices):
    """
    Сравнение моделей с весами и без весов.
    """
    
    print("=== Training WITHOUT sample weights ===")
    trades_ml_no_weights, clf_no_weights = train_without_weights(trades_df, feats)
    
    print("\n=== Training WITH sample weights ===")
    trades_ml_with_weights, clf_with_weights = train_with_sample_weights(trades_df, feats, close_prices)
    
    # Сравнение результатов
    print("\n=== Comparison ===")
    print("Model without weights:")
    print(classification_report(trades_ml_no_weights["y_true"], trades_ml_no_weights["y_pred"]))
    
    print("\nModel with weights:")
    print(classification_report(trades_ml_with_weights["y_true"], trades_ml_with_weights["y_pred"]))
    
    return trades_ml_no_weights, trades_ml_with_weights


def train_without_weights(trades_df, feats):
    """
    Оригинальный код без sample weights.
    """
    
    # Подготовка данных
    trades_long = trades_df[trades_df.direction == 1].dropna(subset=feats)
    trades_short = trades_df[trades_df.direction == -1].dropna(subset=feats)
    
    trades_long["target"] = trades_long["bin"].replace(-1, 0)
    trades_short["target"] = trades_short["bin"].replace(-1, 0).astype(int)
    
    trades_ml = trades_long
    X = trades_ml[feats]
    y = trades_ml['target']
    
    # Инициализация модели
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
    
    y_true_all = np.zeros_like(y)
    y_pred_all = np.zeros_like(y)
    all_val_idx = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        clf.fit(X_train, y_train)  # Без sample_weight
        y_pred = clf.predict(X_val)
        
        y_true_all[val_idx] = y_val
        y_pred_all[val_idx] = y_pred
        all_val_idx.extend(val_idx)
        
        print(f"Fold {fold+1} — Accuracy: {(y_val == y_pred).mean():.3f}")
    
    print("\n=== Classification Report ===")
    print(classification_report(y.iloc[all_val_idx], y_pred_all[all_val_idx]))
    
    trades_ml["y_true"] = y
    trades_ml["y_pred"] = y_pred_all
    
    return trades_ml, clf


# Пример использования:
if __name__ == "__main__":
    """
    Пример использования с вашими данными:
    
    # Предполагаем, что у вас есть:
    # trades - словарь с трейдами по символам
    # feats - список фич
    # close_prices - Series с ценами закрытия
    
    trades_df = trades["BTCUSDT"]
    close_prices = get_close_prices("BTCUSDT")  # ваша функция получения цен
    
    # Обучить с весами
    trades_ml_with_weights, model = train_with_sample_weights(trades_df, feats, close_prices)
    
    # Или сравнить с и без весов
    trades_no_weights, trades_with_weights = compare_with_and_without_weights(trades_df, feats, close_prices)
    """
    
    print("Sample weights integration example")
    print("Use train_with_sample_weights() function with your data") 