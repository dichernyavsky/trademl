import pandas as pd
from datetime import datetime, timedelta
import os
import glob
from pathlib import Path
from typing import List, Dict, Optional, Union

class DataLoader:
    """
    Class for loading historical market data.
    Supports loading from various sources like CSV files, Yahoo Finance, etc.
    """
    
    @staticmethod
    def load_ohlcv_from_csv(file_path, date_column=None, normalize_columns_names=True):
        """
        Load data from CSV file.
        
        Args:
            file_path (str): Path to the CSV file.
            date_column (str, optional): Name of the column containing dates. 
                                         Will try to detect 'Date', 'date', 'Time', 'time', etc. if None.
            normalize_columns (bool): Whether to normalize column names to standard format.
            
        Returns:
            pandas.DataFrame: DataFrame with OHLCV data.
        """
        df = pd.read_csv(file_path)
        
        # Normalize column names if requested
        if normalize_columns_names:
            df = DataLoader._normalize_column_names(df)
        
        # Find and convert date column
        if date_column is None:
            date_candidates = ['Date', 'date', 'Time', 'time', 'datetime', 'Datetime', 'timestamp', 'Timestamp']
            for candidate in date_candidates:
                if candidate in df.columns:
                    date_column = candidate
                    break
        
        if date_column is not None and date_column in df.columns:
            # Try to convert to datetime using different formats if needed
            try:
                df[date_column] = pd.to_datetime(df[date_column])
            except:
                # If simple conversion fails, try some common formats
                for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%d-%m-%Y', '%d/%m/%Y', '%m/%d/%Y']:
                    try:
                        df[date_column] = pd.to_datetime(df[date_column], format=fmt)
                        break
                    except:
                        continue
            
            # Set the date column as index
            df.set_index(date_column, inplace=True)
        
        # Ensure required columns exist in standard format
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns after normalization: {missing_columns}")
        
        return df
    
    @staticmethod
    def _normalize_column_names(df):
        """
        Normalize column names to standard format (Open, High, Low, Close, Volume).
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data.
            
        Returns:
            pandas.DataFrame: DataFrame with normalized column names.
        """
        column_map = {
            'open': 'Open', 'Open': 'Open', 'OPEN': 'Open',
            'high': 'High', 'High': 'High', 'HIGH': 'High',
            'low': 'Low', 'Low': 'Low', 'LOW': 'Low',
            'close': 'Close', 'Close': 'Close', 'CLOSE': 'Close',
            'volume': 'Volume', 'Volume': 'Volume', 'VOLUME': 'Volume',
            # Add other variations as needed
        }
        
        # Create a map for columns that exist in the DataFrame
        rename_dict = {}
        for col in df.columns:
            if col.lower() in [k.lower() for k in column_map.keys()]:
                # Find the proper standard name
                for original, standard in column_map.items():
                    if col.lower() == original.lower():
                        rename_dict[col] = standard
                        break
        
        # Rename columns based on the map
        return df.rename(columns=rename_dict)
    
    
    @staticmethod
    def load_from_list(basedir, timeframe, tickers=None, normalize_columns_names=True, file_format="csv"):
        """
        Load data for multiple tickers from a structured directory.
        
        Args:
            basedir (str): Base directory containing timeframe subdirectories
            timeframe (str): Timeframe folder name (e.g., '1h', '5m', '1d')
            tickers (list, optional): List of ticker symbols to load. If None, loads all available tickers.
            normalize_columns_names (bool): Whether to normalize column names to standard format.
            file_format (str): File format to load ('csv' or 'parquet')
            
        Returns:
            dict: Dictionary mapping ticker symbols to their corresponding DataFrames
        """
        timeframe_dir = os.path.join(basedir, timeframe)
        
        # Check if timeframe directory exists
        if not os.path.exists(timeframe_dir):
            raise ValueError(f"Timeframe directory not found: {timeframe_dir}")
        
        # If tickers not specified, get all files in the directory
        if tickers is None:
            if file_format.lower() == "parquet":
                files = glob.glob(os.path.join(timeframe_dir, "*.parquet"))
            else:
                files = glob.glob(os.path.join(timeframe_dir, "*.csv"))
            tickers = [os.path.splitext(os.path.basename(file))[0] for file in files]
        
        result = {}
        for ticker in tickers:
            file_path = os.path.join(timeframe_dir, f"{ticker}.{file_format}")
            if os.path.exists(file_path):
                try:
                    if file_format.lower() == "parquet":
                        df = DataLoader.load_ohlcv_from_parquet(file_path, normalize_columns_names=normalize_columns_names)
                    else:
                        df = DataLoader.load_ohlcv_from_csv(file_path, normalize_columns_names=normalize_columns_names)
                    result[ticker] = df
                except Exception as e:
                    print(f"Error loading {ticker}: {str(e)}")
            else:
                print(f"File not found for ticker {ticker}: {file_path}")
        
        return result
    
    @staticmethod
    def load_ohlcv_from_parquet(file_path, normalize_columns_names=True):
        """
        Load data from parquet file.
        
        Args:
            file_path (str): Path to the parquet file.
            normalize_columns_names (bool): Whether to normalize column names to standard format.
            
        Returns:
            pandas.DataFrame: DataFrame with OHLCV data.
        """
        df = pd.read_parquet(file_path)
        
        # Normalize column names if requested
        if normalize_columns_names:
            df = DataLoader._normalize_column_names(df)
        
        # Ensure required columns exist in standard format
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns after normalization: {missing_columns}")
        
        return df
    
    @staticmethod
    def load_crypto_data_single_timeframe(symbols: List[str], 
                                         timeframe: str = "1h",
                                         data_root: str = "data/crypto",
                                         normalize_columns: bool = True,
                                         start_date: Optional[str] = None,
                                         end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Load cryptocurrency data for a single timeframe.
        
        Args:
            symbols: List of trading pair symbols to load
            timeframe: Time interval to load (e.g., "1h", "5m", "1d")
            data_root: Root directory for crypto data
            normalize_columns: Whether to normalize column names
            start_date: Start date for filtering (format: "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS")
            end_date: End date for filtering (format: "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS")
            
        Returns:
            Dictionary: {symbol: DataFrame}
        """
        # Map timeframe to folder name
        timeframe_folders = {
            "1s": "1_second",
            "1m": "1_minute",
            "5m": "5_minutes", 
            "1h": "1_hour",
            "1d": "1_day"
        }
        
        if timeframe not in timeframe_folders:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {list(timeframe_folders.keys())}")
            
        timeframe_folder = timeframe_folders[timeframe]
        timeframe_path = os.path.join(data_root, timeframe_folder)
        
        if not os.path.exists(timeframe_path):
            raise ValueError(f"Directory not found: {timeframe_path}")
        
        # Convert date strings to datetime objects
        start_dt = None
        end_dt = None
        
        if start_date:
            try:
                start_dt = pd.to_datetime(start_date)
                print(f"Filtering data from: {start_dt}")
            except Exception as e:
                raise ValueError(f"Invalid start_date format: {start_date}. Use 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'")
        
        if end_date:
            try:
                end_dt = pd.to_datetime(end_date)
                print(f"Filtering data until: {end_dt}")
            except Exception as e:
                raise ValueError(f"Invalid end_date format: {end_date}. Use 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'")
        
        result = {}
        
        for symbol in symbols:
            file_path = os.path.join(timeframe_path, f"{symbol}.parquet")
            if os.path.exists(file_path):
                try:
                    df = DataLoader.load_ohlcv_from_parquet(file_path, normalize_columns)
                    
                    # Filter by date range if specified
                    if start_dt is not None or end_dt is not None:
                        original_len = len(df)
                        
                        if start_dt is not None:
                            df = df[df.index >= start_dt]
                        
                        if end_dt is not None:
                            df = df[df.index <= end_dt]
                        
                        filtered_len = len(df)
                        print(f"Loaded {symbol} @ {timeframe}: {filtered_len} records (filtered from {original_len})")
                    else:
                        print(f"Loaded {symbol} @ {timeframe}: {len(df)} records")
                    
                    result[symbol] = df
                except Exception as e:
                    print(f"Error loading {symbol} @ {timeframe}: {str(e)}")
            else:
                print(f"File not found: {file_path}")
        
        return result
    
    @staticmethod
    def load_crypto_data(symbols: List[str], 
                        intervals: List[str] = ["1h", "5m", "1d", "1m"],
                        data_root: str = "data/crypto",
                        normalize_columns: bool = True) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load cryptocurrency data from the organized directory structure.
        
        Args:
            symbols: List of trading pair symbols to load
            intervals: List of time intervals to load
            data_root: Root directory for crypto data
            normalize_columns: Whether to normalize column names
            
        Returns:
            Nested dictionary: {interval: {symbol: DataFrame}}
        """
        # Map intervals to folder names
        interval_folders = {
            "1s": "1_second",
            "1m": "1_minute",
            "5m": "5_minutes", 
            "1h": "1_hour",
            "1d": "1_day"
        }
        
        result = {}
        
        for interval in intervals:
            if interval not in interval_folders:
                print(f"Warning: Unsupported interval {interval}, skipping")
                continue
                
            interval_folder = interval_folders[interval]
            interval_path = os.path.join(data_root, interval_folder)
            
            if not os.path.exists(interval_path):
                print(f"Warning: Directory not found {interval_path}, skipping")
                continue
            
            result[interval] = {}
            
            for symbol in symbols:
                file_path = os.path.join(interval_path, f"{symbol}.parquet")
                if os.path.exists(file_path):
                    try:
                        df = DataLoader.load_ohlcv_from_parquet(file_path, normalize_columns)
                        result[interval][symbol] = df
                        print(f"Loaded {symbol} @ {interval}: {len(df)} records")
                    except Exception as e:
                        print(f"Error loading {symbol} @ {interval}: {str(e)}")
                else:
                    print(f"File not found: {file_path}")
        
        return result 