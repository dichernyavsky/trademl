# Multiple Samples для Машинного Обучения - План Реализации

## 1. Проблема и Цель

### Проблема
- Текущая система работает с одним набором данных (один символ, один таймфрейм)
- Нет возможности тестировать модели на множественных выборках
- Отсутствует кросс-валидация по времени и символам
- Нет оценки стабильности модели на разных рынках

### Цель
Создать систему для:
- Обучения на множественных выборках (разные символы, периоды, таймфреймы)
- Кросс-валидации по времени (Time Series CV)
- Кросс-валидации по символам (Symbol CV)
- Оценки стабильности модели
- Сравнения производительности на разных рынках

## 2. Архитектура Multiple Samples

### 2.1 Структура данных
```python
# Multiple Samples Dataset
{
    'train': {
        'BTCUSDT': DataFrameTrades,  # Тренировочные данные
        'ETHUSDT': DataFrameTrades,
        'ADAUSDT': DataFrameTrades,
        # ...
    },
    'validation': {
        'BTCUSDT': DataFrameTrades,  # Валидационные данные
        'ETHUSDT': DataFrameTrades,
        # ...
    },
    'test': {
        'BTCUSDT': DataFrameTrades,  # Тестовые данные
        'ETHUSDT': DataFrameTrades,
        # ...
    }
}
```

### 2.2 Классы для Multiple Samples

#### 2.2.1 MultipleSamplesDataset
```python
class MultipleSamplesDataset:
    """
    Управляет множественными выборками для ML.
    """
    
    def __init__(self, data_structure='symbols'):
        """
        Args:
            data_structure: 'symbols' | 'time_periods' | 'timeframes'
        """
        self.data_structure = data_structure
        self.samples = {}  # {sample_id: DataFrameTrades}
        self.metadata = {}  # {sample_id: metadata}
    
    def add_sample(self, sample_id: str, trades_df: pd.DataFrame, metadata: dict = None):
        """Добавить выборку"""
        
    def split_samples(self, split_method: str, **kwargs):
        """Разделить выборки на train/val/test"""
        
    def get_sample(self, sample_id: str) -> pd.DataFrame:
        """Получить конкретную выборку"""
        
    def get_all_samples(self) -> Dict[str, pd.DataFrame]:
        """Получить все выборки"""
```

#### 2.2.2 MultipleSamplesTrainer
```python
class MultipleSamplesTrainer:
    """
    Обучает модели на множественных выборках.
    """
    
    def __init__(self, model_class, feature_engineer=None):
        self.model_class = model_class
        self.feature_engineer = feature_engineer
        self.models = {}  # {sample_id: trained_model}
        self.results = {}  # {sample_id: training_results}
    
    def train_on_samples(self, dataset: MultipleSamplesDataset, 
                        train_samples: List[str], **kwargs):
        """Обучить модель на указанных выборках"""
        
    def cross_validate(self, dataset: MultipleSamplesDataset, 
                      cv_method: str, **kwargs):
        """Кросс-валидация по выборкам"""
        
    def ensemble_predict(self, dataset: MultipleSamplesDataset, 
                        sample_ids: List[str]) -> np.ndarray:
        """Ансамблевые предсказания от нескольких моделей"""
```

#### 2.2.3 MultipleSamplesEvaluator
```python
class MultipleSamplesEvaluator:
    """
    Оценивает производительность на множественных выборках.
    """
    
    def __init__(self):
        self.evaluation_results = {}
    
    def evaluate_model(self, model, dataset: MultipleSamplesDataset, 
                      test_samples: List[str]) -> Dict:
        """Оценить модель на тестовых выборках"""
        
    def compare_models(self, models: Dict[str, BaseModel], 
                      dataset: MultipleSamplesDataset) -> pd.DataFrame:
        """Сравнить несколько моделей"""
        
    def stability_analysis(self, results: Dict) -> Dict:
        """Анализ стабильности модели"""
```

## 3. Методы Кросс-валидации

### 3.1 Time Series Cross-Validation
```python
class TimeSeriesCV:
    """
    Кросс-валидация по времени для временных рядов.
    """
    
    def __init__(self, n_splits=5, test_size=0.2):
        self.n_splits = n_splits
        self.test_size = test_size
    
    def split(self, data: pd.DataFrame) -> List[Tuple]:
        """Разделить данные по времени"""
        # Использует ExpandingWindowSplit или RollingWindowSplit
```

### 3.2 Symbol Cross-Validation
```python
class SymbolCV:
    """
    Кросс-валидация по символам.
    """
    
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
    
    def split(self, samples: Dict[str, pd.DataFrame]) -> List[Tuple]:
        """Разделить данные по символам"""
        # Обучаем на N-1 символах, тестируем на 1 символе
```

### 3.3 Timeframe Cross-Validation
```python
class TimeframeCV:
    """
    Кросс-валидация по таймфреймам.
    """
    
    def split(self, samples: Dict[str, pd.DataFrame]) -> List[Tuple]:
        """Разделить данные по таймфреймам"""
        # Обучаем на одних таймфреймах, тестируем на других
```

## 4. Стратегии Обучения

### 4.1 Single Model per Sample
```python
# Обучаем отдельную модель для каждой выборки
for sample_id in train_samples:
    model = ModelClass()
    model.fit(dataset.get_sample(sample_id))
    models[sample_id] = model
```

### 4.2 Universal Model
```python
# Обучаем одну модель на всех выборках
all_data = pd.concat([dataset.get_sample(sid) for sid in train_samples])
universal_model = ModelClass()
universal_model.fit(all_data)
```

### 4.3 Ensemble Approach
```python
# Обучаем несколько моделей и объединяем предсказания
for sample_id in train_samples:
    model = ModelClass()
    model.fit(dataset.get_sample(sample_id))
    models[sample_id] = model

# Ансамблевые предсказания
predictions = []
for model in models.values():
    pred = model.predict(test_data)
    predictions.append(pred)
ensemble_pred = np.mean(predictions, axis=0)
```

## 5. Метрики для Multiple Samples

### 5.1 Индивидуальные метрики
- Accuracy, Precision, Recall, F1 для каждой выборки
- Sharpe Ratio, Returns, Drawdown для каждой выборки

### 5.2 Агрегированные метрики
- Mean/Median метрик по всем выборкам
- Standard Deviation метрик (стабильность)
- Min/Max метрик (худший/лучший случай)

### 5.3 Стабильность
- Coefficient of Variation (CV = std/mean)
- Rank Correlation между выборками
- Consistency Score

## 6. Примеры Использования

### 6.1 Обучение на множественных символах
```python
# Создаем датасет
dataset = MultipleSamplesDataset()
for symbol in ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']:
    trades = strategy.generate_trades(data[symbol])
    dataset.add_sample(symbol, trades)

# Обучаем модель
trainer = MultipleSamplesTrainer(RandomForestModel)
trainer.train_on_samples(dataset, ['BTCUSDT', 'ETHUSDT'])

# Оцениваем
evaluator = MultipleSamplesEvaluator()
results = evaluator.evaluate_model(trainer.models['BTCUSDT'], dataset, ['ADAUSDT'])
```

### 6.2 Кросс-валидация по времени
```python
# Разделяем данные по времени
tscv = TimeSeriesCV(n_splits=5)
for train_idx, test_idx in tscv.split(data):
    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]
    
    # Генерируем trades
    train_trades = strategy.generate_trades(train_data)
    test_trades = strategy.generate_trades(test_data)
    
    # Обучаем и тестируем
    model.fit(train_trades)
    predictions = model.predict(test_trades)
```

### 6.3 Сравнение стратегий
```python
# Сравниваем разные стратегии на множественных выборках
strategies = [Strategy1(), Strategy2(), Strategy3()]
results = {}

for strategy in strategies:
    trainer = MultipleSamplesTrainer(RandomForestModel)
    # ... обучение ...
    results[strategy.name] = evaluator.evaluate_model(...)

comparison = evaluator.compare_models(results, dataset)
```

## 7. Интеграция с Существующей Системой

### 7.1 Обновление FeatureEngineer
```python
class FeatureEngineer:
    def generate_features_multiple_samples(self, 
                                         original_data: Dict[str, pd.DataFrame],
                                         trades_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Генерировать признаки для множественных выборок"""
```

### 7.2 Обновление BaseModel
```python
class BaseModel:
    def fit_multiple_samples(self, 
                           trades_data: Dict[str, pd.DataFrame],
                           **kwargs) -> 'BaseModel':
        """Обучить модель на множественных выборках"""
```

## 8. План Реализации

### Этап 1: Базовая инфраструктура
1. Создать `MultipleSamplesDataset`
2. Создать `MultipleSamplesTrainer`
3. Создать `MultipleSamplesEvaluator`

### Этап 2: Кросс-валидация
1. Реализовать `TimeSeriesCV`
2. Реализовать `SymbolCV`
3. Реализовать `TimeframeCV`

### Этап 3: Стратегии обучения
1. Single Model per Sample
2. Universal Model
3. Ensemble Approach

### Этап 4: Метрики и анализ
1. Индивидуальные метрики
2. Агрегированные метрики
3. Анализ стабильности

### Этап 5: Интеграция
1. Обновить существующие классы
2. Создать примеры использования
3. Документация и тесты

## 9. Преимущества

1. **Надежность**: Модели тестируются на разных рынках
2. **Стабильность**: Оценка консистентности результатов
3. **Робастность**: Выявление переобучения
4. **Сравнение**: Возможность сравнения стратегий
5. **Ансамбли**: Улучшение предсказаний через ансамбли

## 10. Следующие шаги

1. Начать с реализации `MultipleSamplesDataset`
2. Создать простой пример с двумя символами
3. Добавить базовую кросс-валидацию
4. Постепенно расширять функциональность 