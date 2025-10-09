import os
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Optional, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryptoDataLoader:
    """
    Data loader for cryptocurrency data from Binance API.
    
    Downloads and saves OHLCV data in parquet format with organized directory structure.
    """
    
    def __init__(self, 
                 save_root: str = "data/crypto",
                 api_url: str = "https://api.binance.com/api/v3/klines",
                 request_limit: int = 1000,
                 request_delay: float = 0.5):
        """
        Initialize the crypto data loader.
        
        Args:
            save_root: Root directory for saving data
            api_url: Binance API endpoint for klines
            request_limit: Maximum number of candles per request
            request_delay: Delay between requests to avoid rate limiting
        """
        self.save_root = Path(save_root)
        self.api_url = api_url
        self.request_limit = request_limit
        self.request_delay = request_delay
        
        # Define supported intervals and their folder names
        self.intervals = {
            "1m": "1_minute",
            "5m": "5_minutes", 
            "1h": "1_hour",
            "1d": "1_day"
        }
        
        # Column names for Binance klines data
        self.columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
        ]
        
        # Don't create directories automatically - create them only when needed
    
    def _ensure_directories(self):
        """Create necessary directories for data storage."""
        for interval_folder in self.intervals.values():
            (self.save_root / interval_folder).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directories exist under {self.save_root}")
    
    def get_klines(self, 
                   symbol: str, 
                   interval: str, 
                   start_date: str,
                   end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Download klines data for a specific symbol and interval.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Time interval ('1m', '5m', '1h', '1d')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (optional)
            
        Returns:
            DataFrame with OHLCV data
        """
        if interval not in self.intervals:
            raise ValueError(f"Unsupported interval: {interval}. Supported: {list(self.intervals.keys())}")
        
        # Convert dates to timestamps
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = None
        if end_date:
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
        
        all_data = []
        current_start_ts = start_ts
        
        logger.info(f"Downloading {symbol} @ {interval} from {start_date}")
        
        while True:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start_ts,
                "limit": self.request_limit
            }
            
            if end_ts:
                params["endTime"] = end_ts
            
            try:
                response = requests.get(self.api_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                # Check for API errors
                if isinstance(data, dict) and 'code' in data:
                    logger.error(f"API Error for {symbol}: {data}")
                    break
                
                if not data:
                    logger.info(f"No more data for {symbol}")
                    break
                
                all_data.extend(data)
                
                # Check if we've reached the end
                if len(data) < self.request_limit:
                    break
                
                # Update start time for next request
                last_ts = data[-1][0]
                current_start_ts = last_ts + 1  # Avoid overlap
                
                # Check if we've reached the end date
                if end_ts and current_start_ts >= end_ts:
                    break
                
                # Rate limiting
                time.sleep(self.request_delay)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed for {symbol}: {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected error for {symbol}: {e}")
                break
        
        if not all_data:
            logger.warning(f"No data retrieved for {symbol}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data, columns=self.columns)
        
        # Convert timestamps to datetime
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
        
        # Set open_time as index
        df.set_index("open_time", inplace=True)
        
        # Convert price and volume columns to float32 for efficiency
        numeric_columns = ["open", "high", "low", "close", "volume"]
        df[numeric_columns] = df[numeric_columns].astype("float32")
        
        # Convert trade count to int
        df["number_of_trades"] = df["number_of_trades"].astype("int32")
        
        # Remove unnecessary columns
        df.drop(["close_time", "quote_asset_volume", "taker_buy_base_volume", 
                "taker_buy_quote_volume", "ignore"], axis=1, inplace=True)
        
        logger.info(f"Downloaded {len(df)} candles for {symbol} @ {interval}")
        return df
    
    def save_to_parquet(self, df: pd.DataFrame, symbol: str, interval: str):
        """
        Save DataFrame to parquet file.
        
        Args:
            df: DataFrame to save
            symbol: Trading pair symbol
            interval: Time interval
        """
        if df.empty:
            logger.warning(f"No data to save for {symbol} @ {interval}")
            return
        
        interval_folder = self.intervals[interval]
        file_path = self.save_root / interval_folder / f"{symbol}.parquet"
        
        # Create directory only when saving
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            df.to_parquet(file_path, compression='snappy')
            logger.info(f"Saved {len(df)} records to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save {symbol} @ {interval}: {e}")
    
    def load_from_parquet(self, symbol: str, interval: str) -> pd.DataFrame:
        """
        Load DataFrame from parquet file.
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            
        Returns:
            DataFrame with OHLCV data
        """
        interval_folder = self.intervals[interval]
        file_path = self.save_root / interval_folder / f"{symbol}.parquet"
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return pd.DataFrame()
        
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded {len(df)} records from {file_path}")
            
            # Save OpenTime if index is datetime and OpenTime column doesn't exist
            if 'OpenTime' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
                df['OpenTime'] = df.index
            
            # Set UniqueBarID as index
            df.index = range(1, len(df) + 1)
            df.index.name = 'UniqueBarID'
            
            return df
        except Exception as e:
            logger.error(f"Failed to load {symbol} @ {interval}: {e}")
            return pd.DataFrame()
    
    def download_symbol(self, 
                       symbol: str, 
                       interval: str, 
                       start_date: str,
                       end_date: Optional[str] = None,
                       overwrite: bool = False) -> bool:
        """
        Download and save data for a single symbol.
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            start_date: Start date
            end_date: End date (optional)
            overwrite: Whether to overwrite existing file
            
        Returns:
            True if successful, False otherwise
        """
        interval_folder = self.intervals[interval]
        file_path = self.save_root / interval_folder / f"{symbol}.parquet"
        
        # Check if file exists and overwrite flag
        if file_path.exists() and not overwrite:
            logger.info(f"File exists for {symbol} @ {interval}, skipping (use overwrite=True to force)")
            return True
        
        try:
            df = self.get_klines(symbol, interval, start_date, end_date)
            if not df.empty:
                self.save_to_parquet(df, symbol, interval)
                return True
            else:
                logger.warning(f"No data downloaded for {symbol} @ {interval}")
                return False
        except Exception as e:
            logger.error(f"Failed to download {symbol} @ {interval}: {e}")
            return False
    
    def download_multiple_symbols(self, 
                                 symbols: List[str], 
                                 intervals: List[str],
                                 start_date: str,
                                 end_date: Optional[str] = None,
                                 overwrite: bool = False) -> Dict[str, Dict[str, bool]]:
        """
        Download data for multiple symbols and intervals.
        
        Args:
            symbols: List of trading pair symbols
            intervals: List of time intervals
            start_date: Start date
            end_date: End date (optional)
            overwrite: Whether to overwrite existing files
            
        Returns:
            Dictionary with download results
        """
        results = {}
        
        for interval in intervals:
            if interval not in self.intervals:
                logger.warning(f"Skipping unsupported interval: {interval}")
                continue
                
            results[interval] = {}
            
            for symbol in symbols:
                logger.info(f"Processing {symbol} @ {interval}")
                success = self.download_symbol(symbol, interval, start_date, end_date, overwrite)
                results[interval][symbol] = success
        
        return results
    
    def download_symbol_interval(self, symbol: str, interval: str, start_date: str, end_date: Optional[str] = None, overwrite: bool = False) -> bool:
        """
        Download and save data for a single symbol and interval (for parallel use).
        Returns True if successful, False otherwise.
        """
        try:
            df = self.get_klines(symbol, interval, start_date, end_date)
            if not df.empty:
                self.save_to_parquet(df, symbol, interval)
                return True
            else:
                return False
        except Exception as e:
            logging.error(f"Failed to download {symbol} @ {interval}: {e}")
            return False

    def download_multiple_symbols_parallel(self, symbols: List[str], intervals: List[str], start_date: str, end_date: Optional[str] = None, overwrite: bool = False, max_workers: int = 8) -> Dict[str, Dict[str, bool]]:
        """
        Download data for multiple symbols and intervals in parallel.
        Returns a nested dict: {interval: {symbol: success}}
        """
        results = {interval: {} for interval in intervals}
        tasks = []
        
        # Create all tasks
        for interval in intervals:
            for symbol in symbols:
                tasks.append((symbol, interval))
        
        total_tasks = len(tasks)
        logger.info(f"Starting parallel download of {total_tasks} tasks with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self.download_symbol_interval, symbol, interval, start_date, end_date, overwrite): (symbol, interval)
                for symbol, interval in tasks
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=total_tasks, desc="Downloading", unit="task", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar:
                for future in as_completed(future_to_task):
                    symbol, interval = future_to_task[future]
                    try:
                        success = future.result()
                        status = "✓" if success else "✗"
                        pbar.set_postfix_str(f"{symbol}@{interval} {status}")
                    except Exception as e:
                        logging.error(f"Exception for {symbol} @ {interval}: {e}")
                        success = False
                        pbar.set_postfix_str(f"{symbol}@{interval} ✗")
                    
                    results[interval][symbol] = success
                    pbar.update(1)
        
        return results
    
    def get_available_symbols(self, interval: str) -> List[str]:
        """
        Get list of available symbols for a given interval.
        
        Args:
            interval: Time interval
            
        Returns:
            List of available symbol names
        """
        interval_folder = self.intervals[interval]
        folder_path = self.save_root / interval_folder
        
        if not folder_path.exists():
            return []
        
        symbols = []
        for file_path in folder_path.glob("*.parquet"):
            symbols.append(file_path.stem)
        
        return sorted(symbols)
    
    def get_data_info(self, symbol: str, interval: str) -> Dict:
        """
        Get information about stored data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            
        Returns:
            Dictionary with data information
        """
        df = self.load_from_parquet(symbol, interval)
        
        if df.empty:
            return {"error": "No data found"}
        
        info = {
            "symbol": symbol,
            "interval": interval,
            "start_date": df.index.min().strftime("%Y-%m-%d"),
            "end_date": df.index.max().strftime("%Y-%m-%d"),
            "total_candles": len(df),
            "columns": list(df.columns),
            "file_size_mb": (self.save_root / self.intervals[interval] / f"{symbol}.parquet").stat().st_size / (1024 * 1024)
        }
        
        return info


# Convenience functions for quick usage
def download_crypto_data(symbols: List[str], 
                        intervals: List[str] = ["1h", "5m", "1d", "1m"],
                        start_date: str = "2024-01-01",
                        end_date: Optional[str] = None,
                        save_root: str = "data/crypto",
                        overwrite: bool = False) -> Dict[str, Dict[str, bool]]:
    """
    Convenience function to download crypto data.
    
    Args:
        symbols: List of trading pair symbols
        intervals: List of time intervals
        start_date: Start date
        end_date: End date (optional)
        save_root: Root directory for saving data
        overwrite: Whether to overwrite existing files
        
    Returns:
        Dictionary with download results
    """
    loader = CryptoDataLoader(save_root=save_root)
    return loader.download_multiple_symbols(symbols, intervals, start_date, end_date, overwrite)


def load_crypto_data(symbol: str, 
                    interval: str, 
                    save_root: str = "data/crypto") -> pd.DataFrame:
    """
    Convenience function to load crypto data.
    
    Args:
        symbol: Trading pair symbol
        interval: Time interval
        save_root: Root directory for data
        
    Returns:
        DataFrame with OHLCV data
    """
    loader = CryptoDataLoader(save_root=save_root)
    return loader.load_from_parquet(symbol, interval)


def download_crypto_data_parallel(symbols: List[str], 
                        intervals: List[str] = ["1h", "5m", "1d", "1m"],
                        start_date: str = "2024-01-01",
                        end_date: Optional[str] = None,
                        save_root: str = "data/crypto",
                        overwrite: bool = False,
                        max_workers: int = 8) -> Dict[str, Dict[str, bool]]:
    """
    Parallel version of download_crypto_data.
    """
    loader = CryptoDataLoader(save_root=save_root)
    return loader.download_multiple_symbols_parallel(symbols, intervals, start_date, end_date, overwrite, max_workers)


if __name__ == "__main__":
    # Example usage
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    intervals = ["1h", "5m", "1d", "1m"]
    
    # Download data
    results = download_crypto_data(symbols, intervals, start_date="2024-01-01")
    
    # Print results
    for interval, symbol_results in results.items():
        print(f"\n{interval}:")
        for symbol, success in symbol_results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {symbol}") 