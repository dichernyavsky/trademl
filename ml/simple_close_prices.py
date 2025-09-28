# === ПРОСТОЙ КОД ДЛЯ ПОЛУЧЕНИЯ close_prices ===

# Вариант 1: Самый простой - использовать entry_price
close_prices = trades_long['entry_price']

# Вариант 2: Создать из entry_price и exit_price (более точно)
def create_close_prices_simple(trades_long):
    """Создать close_prices из entry_price и exit_price"""
    
    # Получаем все временные метки
    entry_times = trades_long.index
    exit_times = pd.to_datetime(trades_long['exit_time'])
    
    # Создаем Series с ценами
    prices = pd.Series(dtype=float)
    
    # Добавляем entry prices
    prices.loc[entry_times] = trades_long['entry_price'].values
    
    # Добавляем exit prices
    prices.loc[exit_times] = trades_long['exit_price'].values
    
    # Сортируем по времени
    prices = prices.sort_index()
    
    # Заполняем пропуски интерполяцией
    prices = prices.interpolate(method='linear')
    
    return prices

# Использование:
close_prices = create_close_prices_simple(trades_long)

# Теперь можно использовать sample weights:
from ml.sample_weights import get_sample_weights_for_ml

sample_weights = get_sample_weights_for_ml(
    events=trades_long,
    close=close_prices,
    weight_type='w_event',
    t1_col='exit_time'
)

print(f"Sample weights: min={sample_weights.min():.4f}, max={sample_weights.max():.4f}, mean={sample_weights.mean():.4f}") 