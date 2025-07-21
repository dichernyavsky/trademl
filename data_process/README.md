# Data Processing Module

This module provides functionality for downloading, loading, and managing cryptocurrency data from Binance API.

## Features

- **CryptoDataLoader**: Downloads cryptocurrency data from Binance API
- **DataLoader**: General data loading utilities for various formats
- **Parquet Support**: Efficient data storage and loading using parquet format
- **Organized Structure**: Data stored in organized directory structure
- **Date Range Filtering**: Load data for specific time periods
- **Multiple Data Structures**: Support for both single timeframe and multi-timeframe loading

## Directory Structure

```
data/
└── crypto/
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

## Usage

### Downloading Data

```python
from data_process.crypto_data_loader import download_crypto_data

# Download data for multiple symbols and intervals
results = download_crypto_data(
    symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    intervals=["1h", "5m", "1d", "1m"],
    start_date="2024-01-01",
    end_date="2024-12-01",
    overwrite=False
)
```

### Loading Data - Single Timeframe (Recommended)

```python
from data_process import DataLoader

# Load data for a single timeframe (simplified structure)
dfs = DataLoader.load_crypto_data_single_timeframe(
    symbols=["BTCUSDT", "ETHUSDT"],
    timeframe="1h",
    data_root="data/crypto"
)

# Result: {symbol: DataFrame}
# Example: {'BTCUSDT': DataFrame, 'ETHUSDT': DataFrame}
```

### Loading Data with Date Filtering

```python
from data_process import DataLoader

# Load data for specific date range
dfs = DataLoader.load_crypto_data_single_timeframe(
    symbols=["BTCUSDT"],
    timeframe="1h",
    data_root="data/crypto",
    start_date="2024-11-01",
    end_date="2024-12-01"
)

# Load data from specific date onwards
dfs = DataLoader.load_crypto_data_single_timeframe(
    symbols=["BTCUSDT", "ETHUSDT"],
    timeframe="1h",
    data_root="data/crypto",
    start_date="2024-06-01"
)

# Load data until specific date
dfs = DataLoader.load_crypto_data_single_timeframe(
    symbols=["BTCUSDT"],
    timeframe="1h",
    data_root="data/crypto",
    end_date="2024-03-01"
)
```

### Loading Data - Multiple Timeframes (Legacy)

```python
from data_process import DataLoader

# Load data for multiple timeframes (legacy structure)
dfs = DataLoader.load_crypto_data(
    symbols=["BTCUSDT", "ETHUSDT"],
    intervals=["1h", "5m"],
    data_root="data/crypto"
)

# Result: {interval: {symbol: DataFrame}}
# Example: {'1h': {'BTCUSDT': DataFrame}, '5m': {'BTCUSDT': DataFrame}}
```

### Loading Individual Files

```python
from data_process import DataLoader

# Load single CSV file
df = DataLoader.load_ohlcv_from_csv(
    file_path='data/crypto/1h/BTCUSDT.csv',
    normalize_columns_names=True
)

# Load single parquet file
df = DataLoader.load_ohlcv_from_parquet(
    file_path='data/crypto/1h/BTCUSDT.parquet',
    normalize_columns_names=True
)

# Load multiple files from directory
dfs = DataLoader.load_from_list(
    basedir='data/crypto',
    timeframe='1h',
    tickers=['BTCUSDT', 'ETHUSDT'],
    normalize_columns_names=True
)
```

### Using CryptoDataLoader Class

```python
from data_process.crypto_data_loader import CryptoDataLoader

# Initialize loader
loader = CryptoDataLoader(save_root="data/crypto")

# Download single symbol
success = loader.download_symbol(
    symbol="BTCUSDT",
    interval="1h",
    start_date="2024-01-01",
    end_date="2024-12-01"
)

# Load data
df = loader.load_from_parquet("BTCUSDT", "1h")

# Get data info
info = loader.get_data_info("BTCUSDT", "1h")
print(f"Data range: {info['start_date']} to {info['end_date']}")
print(f"Total candles: {info['total_candles']}")
```

## Data Format

The downloaded data includes the following columns:

- **Open**: Opening price
- **High**: Highest price during the period
- **Low**: Lowest price during the period
- **Close**: Closing price
- **Volume**: Trading volume
- **close_time**: Close time timestamp
- **quote_asset_volume**: Quote asset volume
- **number_of_trades**: Number of trades during the period
- **taker_buy_base_volume**: Taker buy base asset volume
- **taker_buy_quote_volume**: Taker buy quote asset volume

## Supported Timeframes

- `1m`: 1 minute
- `5m`: 5 minutes
- `1h`: 1 hour
- `1d`: 1 day

## Date Format Support

When using date filtering, the following formats are supported:

- `"2024-12-01"` - Date only
- `"2024-12-01 14:30:00"` - Date with time
- `"2024-12-01T14:30:00"` - ISO format

## Data Structures

### Single Timeframe Structure (Recommended)
```python
{
    'BTCUSDT': DataFrame,
    'ETHUSDT': DataFrame,
    ...
}
```

### Multi-Timeframe Structure (Legacy)
```python
{
    '1h': {
        'BTCUSDT': DataFrame,
        'ETHUSDT': DataFrame,
        ...
    },
    '5m': {
        'BTCUSDT': DataFrame,
        'ETHUSDT': DataFrame,
        ...
    }
}
```

## API Rate Limiting

The module includes built-in rate limiting to avoid API restrictions:
- Default delay between requests: 0.5 seconds
- Configurable via `request_delay` parameter
- Automatic error handling for API limits

## Error Handling

The module includes comprehensive error handling:
- Network request failures
- API errors and rate limiting
- File I/O errors
- Data validation
- Invalid date formats

## Performance

- Uses parquet format for efficient storage and loading
- Float32 data types for price columns to reduce memory usage
- Compressed storage with snappy compression
- Indexed by timestamp for fast lookups
- Date filtering reduces memory usage for large datasets

## Scripts

The module includes several utility scripts:

- `download_full_history.py`: Download full historical data for top USDT pairs (parallel)
- `get_popular_symbols.py`: Get and save popular cryptocurrency symbols

## Examples

### Quick Testing with Small Dataset
```python
from data_process import DataLoader
from indicators import SimpleSupportResistance, BollingerBands
from strategies import SimpleSRStrategy

# Load small dataset for quick testing
dfs = DataLoader.load_crypto_data_single_timeframe(
    symbols=["BTCUSDT"],
    timeframe="1h",
    data_root="data/crypto",
    start_date="2024-12-01",
    end_date="2024-12-08"
)

# Apply indicators
indicators = [
    SimpleSupportResistance(lookback=20),
    BollingerBands(window=20, num_std=2) 
]

for indicator in indicators:
    dfs = indicator.calculate(dfs, append=True)

# Test strategy
strategy = SimpleSRStrategy(params={
    'lookback': 20,
    'hold_periods': 50,
    'barrier_method': 'simple',
    'window': 20,
    'multiplier': [2, 2],
    'min_ret': 0.001
})

events = strategy.generate_events(dfs, set_barriers=True)
trades = strategy.generate_trades(dfs)

print(f"Events: {len(events['BTCUSDT'])}")
print(f"Trades: {len(trades['BTCUSDT'])}")
```

See the following files for complete examples:
- `download_full_history.py`: Full historical data download script
- `get_popular_symbols.py`: Symbol fetching script 
- `USAGE_EXAMPLES.md`: Detailed usage examples 