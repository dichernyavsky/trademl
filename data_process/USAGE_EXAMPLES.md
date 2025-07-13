# Data Downloader Usage Examples

Этот файл содержит примеры использования скриптов для загрузки криптовалютных данных.

## 🚀 Быстрый старт

### 1. Получить популярные символы
```bash
# Из корня проекта
python data_process/get_popular_symbols.py

# Из папки data_process
cd data_process
python get_popular_symbols.py
```

**Результат:** Создаст файлы:
- `data/popular_symbols.txt` - 18 популярных символов
- `data/usdt_pairs.txt` - все 403 USDT пары
- `data/binance_symbols.csv` - полный список символов

### 2. Загрузить полную историю для топ-20 пар
```bash
# Из корня проекта
python data_process/download_full_history.py --count 20 --workers 8

# Из папки data_process
cd data_process
python download_full_history.py --count 20 --workers 8
```

**Результат:** Загрузит данные для топ-20 USDT пар с 2017 года по сегодня.

## 📊 Подробные примеры

### Загрузка данных

#### Пример 1: Только оценка времени (без загрузки)
```bash
python data_process/download_full_history.py --estimate-only --count 10
```
```
🚀 Full Historical Data Download (Parallel)
==================================================
📊 Symbols: ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT', 'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT']
📅 Date range: 2017-08-01 to 2025-07-12
⏰ Intervals: ['1m', '5m', '1h', '1d']
💾 Save root: data/crypto
🔄 Overwrite: False
🧵 Workers: 8
⏱️  Estimated download time: 0h 52m

✅ Estimation complete. Use --estimate-only to see download time without downloading.
```

#### Пример 2: Загрузка только 1-часовых данных
```bash
python data_process/download_full_history.py --count 5 --intervals 1h --workers 4
```

#### Пример 3: Загрузка с перезаписью существующих файлов
```bash
python data_process/download_full_history.py --count 10 --intervals 1h 1d --overwrite --workers 6
```

#### Пример 4: Загрузка за определенный период
```bash
python data_process/download_full_history.py --count 5 --start-date 2024-01-01 --end-date 2024-12-31
```

### Параметры командной строки

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--count` | Количество топ USDT пар | 20 |
| `--intervals` | Временные интервалы | 1m 5m 1h 1d |
| `--start-date` | Дата начала (YYYY-MM-DD) | 2017-08-01 |
| `--end-date` | Дата окончания (YYYY-MM-DD) | сегодня |
| `--overwrite` | Перезаписать существующие файлы | False |
| `--save-root` | Папка для сохранения | data/crypto |
| `--workers` | Количество параллельных потоков | 8 |
| `--estimate-only` | Только оценка времени | False |

## 📁 Структура данных

После загрузки данные сохраняются в следующей структуре:

```
data/crypto/
├── 1_minute/
│   ├── BTCUSDT.parquet
│   ├── ETHUSDT.parquet
│   └── ...
├── 5_minutes/
│   ├── BTCUSDT.parquet
│   ├── ETHUSDT.parquet
│   └── ...
├── 1_hour/
│   ├── BTCUSDT.parquet
│   ├── ETHUSDT.parquet
│   └── ...
└── 1_day/
    ├── BTCUSDT.parquet
    ├── ETHUSDT.parquet
    └── ...
```

## 📈 Загрузка данных в Python

### Загрузка одного символа
```python
from data_process.crypto_data_loader import load_crypto_data

# Загрузить BTCUSDT 1-часовые данные
btc_data = load_crypto_data("BTCUSDT", "1h")
print(f"Загружено {len(btc_data)} записей")
print(f"Период: {btc_data.index.min()} - {btc_data.index.max()}")
```

### Загрузка нескольких символов
```python
from data_process.data_loader import DataLoader

# Загрузить несколько символов
crypto_data = DataLoader.load_crypto_data(
    symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    intervals=["1h", "5m"],
    data_root="data/crypto"
)

# Доступ к данным
btc_1h = crypto_data["1h"]["BTCUSDT"]
eth_5m = crypto_data["5m"]["ETHUSDT"]
```

### Использование CryptoDataLoader класса
```python
from data_process.crypto_data_loader import CryptoDataLoader

# Инициализация
loader = CryptoDataLoader(save_root="data/crypto")

# Загрузка данных
df = loader.load_from_parquet("BTCUSDT", "1h")

# Получение информации о данных
info = loader.get_data_info("BTCUSDT", "1h")
print(f"Символ: {info['symbol']}")
print(f"Интервал: {info['interval']}")
print(f"Период: {info['start_date']} - {info['end_date']}")
print(f"Записей: {info['total_candles']}")
print(f"Размер файла: {info['file_size_mb']:.2f} MB")

# Получение списка доступных символов
available_symbols = loader.get_available_symbols("1h")
print(f"Доступные символы для 1h: {available_symbols}")
```

## ⏱️ Оценка времени загрузки

### Формула расчета:
- **1m**: 1440 свечей в день
- **5m**: 288 свечей в день  
- **1h**: 24 свечи в день
- **1d**: 1 свеча в день

### Примеры времени загрузки:

| Символов | Интервалы | Период | Время (8 workers) |
|----------|-----------|--------|-------------------|
| 5 | 1h | 2017-2025 | ~5 минут |
| 10 | 1h | 2017-2025 | ~10 минут |
| 20 | 1h | 2017-2025 | ~20 минут |
| 10 | все | 2017-2025 | ~1 час |
| 20 | все | 2017-2025 | ~2 часа |

## 🔧 Полезные команды

### Проверка размера данных
```bash
# Размер папки с данными
du -sh data/crypto/

# Количество файлов по интервалам
find data/crypto/1_hour -name "*.parquet" | wc -l
find data/crypto/5_minutes -name "*.parquet" | wc -l
```

### Очистка данных
```bash
# Удалить все данные
rm -rf data/crypto/*/

# Удалить только 1-минутные данные (самые большие)
rm -rf data/crypto/1_minute/
```

### Проверка целостности данных
```python
import pandas as pd
from pathlib import Path

# Проверить все файлы
data_root = Path("data/crypto")
for interval_dir in data_root.iterdir():
    if interval_dir.is_dir():
        print(f"\n{interval_dir.name}:")
        for file in interval_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(file)
                print(f"  ✓ {file.name}: {len(df)} записей")
            except Exception as e:
                print(f"  ✗ {file.name}: ОШИБКА - {e}")
```

## 🚨 Частые проблемы

### 1. "File not found" при загрузке
```python
# Проверьте, что файл существует
from pathlib import Path
file_path = Path("data/crypto/1_hour/BTCUSDT.parquet")
print(f"Файл существует: {file_path.exists()}")
```

### 2. Медленная загрузка
- Увеличьте количество workers: `--workers 12`
- Загружайте только нужные интервалы
- Используйте меньший период времени

### 3. Ошибки API
- Скрипт автоматически обрабатывает ошибки
- Проверьте интернет-соединение
- Подождите между большими загрузками

## 📝 Рекомендации

1. **Начните с малого**: Сначала загрузите 5-10 символов
2. **Используйте оценку времени**: `--estimate-only` перед большими загрузками
3. **Мониторьте место на диске**: 1-минутные данные занимают много места
4. **Сохраняйте логи**: Для отладки проблем
5. **Проверяйте данные**: Всегда проверяйте загруженные данные

## 🎯 Типичные сценарии использования

### Сценарий 1: Быстрый тест
```bash
# Загрузить только BTC и ETH за последний год
python data_process/download_full_history.py \
  --count 2 \
  --intervals 1h \
  --start-date 2024-01-01 \
  --workers 4
```

### Сценарий 2: Полная загрузка
```bash
# Загрузить топ-20 пар за всю историю
python data_process/download_full_history.py \
  --count 20 \
  --intervals 1m 5m 1h 1d \
  --workers 8 \
  --overwrite
```

### Сценарий 3: Обновление данных
```bash
# Обновить данные за последний месяц
python data_process/download_full_history.py \
  --count 10 \
  --start-date 2024-06-01 \
  --overwrite
``` 