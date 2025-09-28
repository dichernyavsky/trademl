# ML Performance Integration

Этот модуль интегрирует машинное обучение с анализом производительности торговых стратегий, позволяя оценивать качество ML-моделей через призму торговых метрик.

## Обзор

Интеграция `MLPerformanceIntegrator` предоставляет:

1. **Конвертация предсказаний в торговые решения** - преобразование выходов ML-моделей в торговые сигналы
2. **Анализ производительности ML-стратегий** - оценка торговых метрик для ML-подходов
3. **Сравнение различных ML-моделей** - сопоставление разных подходов к обучению
4. **Кросс-валидация с торговыми метриками** - оценка стабильности моделей

## Основные компоненты

### MLPerformanceIntegrator

Главный класс для интеграции ML с анализом производительности:

```python
from ml import MLPerformanceIntegrator

integrator = MLPerformanceIntegrator(
    initial_capital=10000.0,
    risk_free_rate=0.02,
    timeframe="1H"
)
```

### Методы анализа

#### 1. Конвертация предсказаний в сделки

```python
# Преобразование предсказаний ML в торговые решения
trades = integrator.convert_predictions_to_trades(
    predictions=predictions,
    dataset=dataset,
    test_samples=test_samples,
    threshold=0.0,  # Порог для принятия решений
    min_confidence=0.0  # Минимальная уверенность
)
```

#### 2. Анализ производительности ML-стратегии

```python
# Полный анализ производительности
performance = integrator.analyze_ml_strategy_performance(
    predictions=predictions,
    dataset=dataset,
    test_samples=test_samples,
    threshold=0.0,
    min_confidence=0.0
)

# Вывод результатов
integrator.print_performance_summary(performance)
```

#### 3. Сравнение моделей

```python
# Сравнение разных ML-подходов
comparison = integrator.compare_ml_models(
    models=models,
    dataset=dataset,
    test_samples=test_samples,
    threshold=0.0,
    min_confidence=0.0
)
```

#### 4. Кросс-валидация с торговыми метриками

```python
# Оценка с использованием кросс-валидации
cv_performance = integrator.evaluate_cross_validation_performance(
    trainer=trainer,
    dataset=dataset,
    train_samples=train_samples,
    test_samples=test_samples,
    training_method='single_models',  # или 'universal_model', 'ensemble'
    threshold=0.0,
    min_confidence=0.0
)
```

## Пример использования

```python
from data_process import DataLoader
from indicators import SimpleSupportResistance
from strategies import SimpleSRStrategy
from ml import (
    MultipleSamplesDataset, 
    MultipleSamplesTrainer, 
    MLPerformanceIntegrator,
    RandomForestModel
)

# 1. Подготовка данных
symbols = ["BTCUSDT", "ETHUSDT"]
dfs = DataLoader.load_crypto_data_single_timeframe(
    symbols=symbols,
    timeframe="1h",
    start_date="2023-01-01",
    end_date="2024-01-01"
)

# Добавление индикаторов
indicators = [SimpleSupportResistance(lookback=20)]
for indicator in indicators:
    dfs = indicator.calculate(dfs, append=True)

# Генерация событий и сделок
strategy = SimpleSRStrategy(params={
    'lookback': 20,
    'hold_periods': 50,
    'barrier_method': 'simple',
    'window': 40,
    'multiplier': [2, 2],
    'min_ret': 0.001
})

events = strategy.generate_events(dfs, set_barriers=True)
trades = strategy.generate_trades(dfs)

# 2. Подготовка ML-датасета
dataset = MultipleSamplesDataset('symbols')
for symbol in dfs.keys():
    if symbol in trades and len(trades[symbol]) > 0:
        trade_data = dfs[symbol].loc[trades[symbol].index]
        trade_data['bin'] = trades[symbol]['bin'].values
        trade_data['entry_price'] = trades[symbol]['entry_price'].values
        trade_data['exit_price'] = trades[symbol]['exit_price'].values
        dataset.add_sample(symbol, trade_data)

# Разделение на train/test
all_samples = list(dataset.get_sample_ids())
train_samples = all_samples[:len(all_samples)//2]
test_samples = all_samples[len(all_samples)//2:]

dataset.splits['default'] = {
    'train': train_samples,
    'val': [],
    'test': test_samples
}

# 3. Обучение и оценка
trainer = MultipleSamplesTrainer(
    model_class=RandomForestModel,
    n_estimators=100,
    random_state=42
)

integrator = MLPerformanceIntegrator(
    initial_capital=10000.0,
    risk_free_rate=0.02,
    timeframe="1H"
)

# Оценка разных подходов
approaches = ['single_models', 'universal_model', 'ensemble']
results = {}

for approach in approaches:
    performance = integrator.evaluate_cross_validation_performance(
        trainer=trainer,
        dataset=dataset,
        train_samples=train_samples,
        test_samples=test_samples,
        training_method=approach,
        threshold=0.0,
        min_confidence=0.0
    )
    results[approach] = performance
    integrator.print_performance_summary(performance)
```

## Метрики производительности

Интеграция рассчитывает следующие торговые метрики:

### Базовые метрики
- **Total Return** - общая доходность
- **Sharpe Ratio** - коэффициент Шарпа
- **Max Drawdown** - максимальная просадка
- **Win Rate** - процент прибыльных сделок
- **Profit Factor** - фактор прибыли
- **Total Trades** - общее количество сделок

### Детальные метрики
- **Annualized Return** - годовая доходность
- **Volatility** - волатильность
- **Average Win/Loss** - средний выигрыш/проигрыш
- **Consecutive Wins/Losses** - максимальные серии выигрышей/проигрышей

### Агрегированные метрики
- **Mean/Std/Min/Max/Median** - статистики по всем сэмплам
- **Stability Metrics** - метрики стабильности

## Подходы к обучению

### 1. Single Models (Отдельные модели)
- Одна модель на каждый символ/период
- Максимальная адаптация к специфике данных
- Может быть нестабильным при малом количестве данных

### 2. Universal Model (Универсальная модель)
- Одна модель на все данные
- Лучшая обобщающая способность
- Может не учитывать специфику отдельных символов

### 3. Ensemble (Ансамбль)
- Комбинация нескольких моделей
- Улучшенная стабильность
- Более сложная интерпретация

## Пороги принятия решений

### Threshold (Порог)
- `threshold = 0.0` - использовать все ненулевые предсказания
- `threshold > 0.0` - использовать только предсказания выше порога
- Помогает фильтровать слабые сигналы

### Min Confidence (Минимальная уверенность)
- Фильтрует сделки по уверенности модели
- Уменьшает количество ложных сигналов
- Может снизить общее количество сделок

## Сохранение результатов

```python
# Сохранение результатов в JSON
integrator.save_performance_results(results, 'ml_performance_results.json')
```

## Преимущества интеграции

1. **Реалистичная оценка** - модели оцениваются через торговые метрики
2. **Сравнение подходов** - возможность сравнить разные ML-стратегии
3. **Стабильность** - оценка через множественные сэмплы
4. **Интерпретируемость** - результаты в понятных торговых терминах
5. **Гибкость** - различные пороги и методы обучения

## Использование в реальной торговле

Интеграция позволяет:

1. **Выбрать лучшую модель** на основе торговых метрик
2. **Оптимизировать пороги** для максимизации прибыли
3. **Оценить стабильность** стратегии на разных периодах
4. **Сравнить подходы** и выбрать наиболее подходящий
5. **Мониторить производительность** в реальном времени

Это создает полный цикл от обучения ML-моделей до оценки их торговой эффективности. 