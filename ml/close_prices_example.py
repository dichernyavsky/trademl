"""
Пример как получить close_prices из существующих данных.
"""

import pandas as pd
import numpy as np

# Вариант 1: Если у вас есть OHLCV данные
def get_close_prices_from_ohlcv(symbol="BTCUSDT"):
    """
    Получить close_prices из OHLCV данных.
    """
    # Предполагаем, что у вас есть функция загрузки OHLCV данных
    # ohlcv_data = load_ohlcv_data(symbol)  # ваша функция загрузки
    
    # Если у вас есть DataFrame с OHLCV данными:
    # ohlcv_data = pd.DataFrame({
    #     'open': [...],
    #     'high': [...],
    #     'low': [...],
    #     'close': [...],
    #     'volume': [...]
    # }, index=timestamps)
    
    # close_prices = ohlcv_data['close']
    # return close_prices
    
    pass

# Вариант 2: Создать close_prices из entry_price и exit_price
def create_close_prices_from_trades(trades_long):
    """
    Создать приблизительные close_prices из entry_price и exit_price трейдов.
    Это не идеально, но может работать для sample weights.
    """
    print("Creating close_prices from trades data...")
    
    # Получаем все уникальные временные метки
    all_times = set()
    
    # Добавляем времена входа
    all_times.update(trades_long.index)
    
    # Добавляем времена выхода
    all_times.update(pd.to_datetime(trades_long['exit_time']))
    
    # Сортируем времена
    all_times = sorted(all_times)
    
    # Создаем DataFrame с временными метками
    close_prices = pd.DataFrame(index=all_times)
    
    # Заполняем цены
    for idx, row in trades_long.iterrows():
        entry_time = idx
        exit_time = pd.to_datetime(row['exit_time'])
        entry_price = row['entry_price']
        exit_price = row['exit_price']
        
        # Устанавливаем цены
        close_prices.loc[entry_time, 'price'] = entry_price
        close_prices.loc[exit_time, 'price'] = exit_price
    
    # Заполняем пропуски интерполяцией
    close_prices = close_prices['price'].interpolate(method='linear')
    
    print(f"Created close_prices with {len(close_prices)} timestamps")
    print(f"Price range: {close_prices.min():.2f} - {close_prices.max():.2f}")
    
    return close_prices

# Вариант 3: Использовать только entry_price (самый простой)
def use_entry_prices_as_close(trades_long):
    """
    Использовать entry_price как close_prices.
    Самый простой вариант, но менее точный.
    """
    print("Using entry_prices as close_prices...")
    
    close_prices = trades_long['entry_price'].copy()
    
    print(f"Using {len(close_prices)} entry prices as close prices")
    print(f"Price range: {close_prices.min():.2f} - {close_prices.max():.2f}")
    
    return close_prices

# Пример использования с вашими данными
def example_usage():
    """
    Пример использования с вашими данными.
    """
    
    # Ваш код подготовки данных
    # trades_df = trades["BTCUSDT"]
    # trades_long = trades_df[trades_df.direction==1].dropna(subset=feats)
    
    # Вариант 1: Если у вас есть OHLCV данные
    # close_prices = get_close_prices_from_ohlcv("BTCUSDT")
    
    # Вариант 2: Создать из трейдов (рекомендуется)
    close_prices = create_close_prices_from_trades(trades_long)
    
    # Вариант 3: Использовать entry_price (самый простой)
    # close_prices = use_entry_prices_as_close(trades_long)
    
    # Теперь можно использовать sample weights
    from ml.sample_weights import get_sample_weights_for_ml
    
    sample_weights = get_sample_weights_for_ml(
        events=trades_long,
        close=close_prices,
        weight_type='w_event',
        t1_col='exit_time'  # у вас уже есть эта колонка
    )
    
    print(f"Sample weights calculated: {len(sample_weights)} weights")
    print(f"Weight stats: min={sample_weights.min():.4f}, max={sample_weights.max():.4f}, mean={sample_weights.mean():.4f}")
    
    return sample_weights, close_prices

# Минимальный код для копирования в Jupyter:
"""
# === МИНИМАЛЬНЫЙ КОД ДЛЯ ПОЛУЧЕНИЯ close_prices ===

# Вариант 1: Создать из трейдов (рекомендуется)
close_prices = create_close_prices_from_trades(trades_long)

# Вариант 2: Использовать entry_price (самый простой)
# close_prices = trades_long['entry_price']

# Теперь можно использовать sample weights
sample_weights = get_sample_weights_for_ml(
    events=trades_long,
    close=close_prices,
    weight_type='w_event',
    t1_col='exit_time'
)
""" 