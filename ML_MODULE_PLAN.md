# План модуля машинного обучения для TradeML

## Обзор архитектуры

Модуль машинного обучения будет построен на основе DataFrameTrades, который содержит:
- **Метки (labels)**: `bin` (1, -1, 0) - результат торговли
- **Признаки (features)**: Все индикаторы на момент входа в позицию
- **Метаданные**: Время входа/выхода, цены, направление и т.д.

## Структура модуля (УПРОЩЕННАЯ)

### 1. Основные компоненты

```
ml/
├── __init__.py
├── models/                    # Модели машинного обучения
│   ├── __init__.py
│   ├── base_model.py         # Базовый класс для всех ML моделей
│   └── random_forest.py      # Случайный лес
├── feature_engineering/       # Инженерия признаков
│   ├── __init__.py
│   └── feature_engineer.py   # Простой Feature Engineer
├── optimization/             # Оптимизация гиперпараметров (TODO)
│   └── __init__.py
├── evaluation/               # Оценка моделей (TODO)
│   └── __init__.py
├── ensemble/                 # Ансамблевые методы (TODO)
│   └── __init__.py
└── utils/                    # Утилиты (TODO)
    └── __init__.py
```

### 2. Классы моделей (models/)

#### BaseModel (base_model.py) ✅ РЕАЛИЗОВАН
```python
class BaseModel:
    def __init__(self, model_type='classification', **kwargs):
        self.model_type = model_type
        self.model = None
        self.feature_columns = []
        self.target_column = 'bin'
        
    def fit(self, trades_df, **kwargs):
        """Обучение модели на DataFrameTrades"""
        
    def predict(self, trades_df):
        """Предсказание на новых данных"""
        
    def predict_proba(self, trades_df):
        """Предсказание вероятностей"""
        
    def save(self, path):
        """Сохранение модели"""
        
    def load(self, path):
        """Загрузка модели"""
```

#### RandomForestModel (random_forest.py) ✅ РЕАЛИЗОВАН
- **RandomForestModel**: Случайный лес для классификации/регрессии

### 3. Инженерия признаков (feature_engineering/)

#### FeatureEngineer (feature_engineer.py) ✅ РЕАЛИЗОВАН
```python
class FeatureEngineer:
    def __init__(self, indicators=None):
        self.indicators = indicators or []
        self.feature_columns = []
        
    def add_indicator(self, indicator):
        """Добавление индикатора для генерации признаков"""
        
    def generate_features(self, original_data, trades_df):
        """
        Основной метод генерации признаков.
        
        Args:
            original_data: Исходный датасет (DataFrame или dict)
            trades_df: DataFrameTrades с точками входа
            
        Returns:
            trades_df: Обогащенный DataFrameTrades с новыми признаками
        """
        # 1. Генерируем признаки для основного датасета
        enriched_data = self._generate_features_for_dataset(original_data)
        
        # 2. Извлекаем значения признаков в точках входа (векторизованно)
        trades_df = self._extract_features_vectorized(trades_df, enriched_data)
        
        return trades_df
```

## Оптимизация производительности 🚀

### Векторизованное решение
Используем pandas merge/join для быстрого извлечения признаков:
```python
# Быстрый подход (O(n log n))
features_df = enriched_data[feature_columns].copy()
result = trades_df.merge(
    features_df, 
    left_index=True, 
    right_index=True, 
    how='left'
)
```

### Ожидаемая производительность
| Размер датасета | Время обработки | Тrades/сек |
|----------------|-----------------|------------|
| 1,000 trades   | 0.001s         | 1,000,000  |
| 10,000 trades  | 0.005s         | 2,000,000  |
| 50,000 trades  | 0.02s          | 2,500,000  |
| 100,000 trades | 0.04s          | 2,500,000  |

### Дополнительные оптимизации (для будущего)

#### 1. Batch Processing
```python
def extract_features_batch(self, trades_df, enriched_data, batch_size=10000):
    """Обработка больших датасетов батчами"""
    results = []
    for i in range(0, len(trades_df), batch_size):
        batch = trades_df.iloc[i:i+batch_size]
        batch_result = self._extract_features_vectorized(batch, enriched_data)
        results.append(batch_result)
    return pd.concat(results)
```

#### 2. Memory-Efficient Processing
```python
def extract_features_memory_efficient(self, trades_df, enriched_data):
    """Эффективное использование памяти"""
    # Выбираем только нужные колонки
    feature_columns = self._get_feature_columns(enriched_data)
    features_df = enriched_data[feature_columns].copy()
    
    # Используем merge с оптимизированными типами данных
    return trades_df.merge(features_df, left_index=True, right_index=True, how='left')
```

## Логика Feature Engineering

### Основной принцип:
1. **Генерация признаков**: Сначала генерируем все признаки для основного датасета используя индикаторы
2. **Извлечение значений**: Затем извлекаем значения признаков в точках входа (trades.index)

### Векторизованная реализация:
```python
def _extract_features_vectorized(self, trades_df, enriched_data):
    """Векторизованное извлечение признаков"""
    # Выбираем только колонки признаков
    exclude_columns = {'Open', 'High', 'Low', 'Close', 'Volume', 'bin', 'direction', ...}
    feature_columns = [col for col in enriched_data.columns 
                      if col not in exclude_columns]
    
    # Быстрый merge по индексу
    features_df = enriched_data[feature_columns].copy()
    result = trades_df.merge(
        features_df, 
        left_index=True, 
        right_index=True, 
        how='left'
    )
    
    return result
```

## Примеры использования

### 1. Простое обучение модели
```python
from ml import RandomForestModel, FeatureEngineer
from indicators import SimpleSupportResistance

# Загружаем данные и генерируем trades
strategy = SimpleSRStrategy(params={...})
trades = strategy.generate_trades(data)

# Инженерия признаков (векторизованно по умолчанию)
feature_engineer = FeatureEngineer()
feature_engineer.add_indicator(SimpleSupportResistance(lookback=20))
trades_with_features = feature_engineer.generate_features(data, trades)

# Обучение модели
model = RandomForestModel(n_estimators=100, max_depth=5)
model.fit(trades_with_features)

# Предсказания
predictions = model.predict(trades_with_features)
```

### 2. Расширение BaseStrategy (будущее)
```python
class MLStrategy(BaseStrategy):
    def __init__(self, ml_model, feature_engineer=None, **kwargs):
        super().__init__(**kwargs)
        self.ml_model = ml_model
        self.feature_engineer = feature_engineer
        
    def generate_trades(self, data, **kwargs):
        # Генерируем базовые trades
        trades = super().generate_trades(data, **kwargs)
        
        # Добавляем ML признаки
        if self.feature_engineer:
            trades = self.add_ml_features(trades, data)
            
        # Добавляем ML предсказания
        if self.ml_model is not None:
            trades = self.add_ml_predictions(trades)
            
        return trades
```

## Что уже реализовано ✅

1. **BaseModel** - базовый класс для всех ML моделей
2. **RandomForestModel** - конкретная реализация случайного леса
3. **FeatureEngineer** - простой инженер признаков на основе индикаторов
4. **Векторизованная оптимизация** - быстрая обработка больших датасетов
5. **Интеграция с существующими индикаторами** - работает с любыми индикаторами из модуля indicators
6. **Пример использования** - ml_example.py
7. **Тест производительности** - test_feature_engineer_performance.py

## Что планируется реализовать 🔄

1. **Дополнительные модели**:
   - XGBoostModel
   - LightGBMModel
   - LogisticRegressionModel

2. **Оптимизация гиперпараметров**:
   - HyperparameterOptimizer с Optuna
   - Байесовская оптимизация

3. **Оценка моделей**:
   - ModelEvaluator
   - Временная кросс-валидация
   - Торговые метрики

4. **Ансамблевые методы**:
   - EnsembleModel
   - Voting и Stacking

5. **Интеграция с BaseStrategy**:
   - MLStrategy класс
   - Автоматическая генерация признаков и предсказаний

6. **Дополнительные оптимизации**:
   - Batch processing для очень больших датасетов
   - Parallel processing
   - Memory-efficient processing

## Зависимости

```python
# requirements.txt additions
scikit-learn>=1.0.0
# xgboost>=1.5.0  # для будущих моделей
# lightgbm>=3.3.0  # для будущих моделей
# optuna>=3.0.0  # для оптимизации
```

## Следующие шаги

1. **Тестирование текущей реализации** на реальных данных
2. **Запуск тестов производительности** для валидации оптимизации
3. **Добавление дополнительных моделей** (XGBoost, LightGBM)
4. **Реализация оптимизации гиперпараметров**
5. **Создание MLStrategy** для интеграции с существующей архитектурой
6. **Добавление оценки моделей** и метрик
7. **Документация и примеры** 