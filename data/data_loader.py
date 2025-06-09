import pandas as pd
from datetime import datetime, timedelta
import os
import glob

class DataLoader:
    """
    Class for loading historical market data.
    Supports loading from various sources like CSV files, Yahoo Finance, etc.
    """
    
    @staticmethod
    def load_ohlcv_from_csv(file_path, date_column=None, normalize_columns=True):
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
        if normalize_columns:
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
    def load_from_list(basedir, timeframe, tickers=None, normalize_columns=True):
        """
        Load data for multiple tickers from a structured directory.
        
        Args:
            basedir (str): Base directory containing timeframe subdirectories
            timeframe (str): Timeframe folder name (e.g., '1h', '5m', '1d')
            tickers (list, optional): List of ticker symbols to load. If None, loads all available tickers.
            normalize_columns (bool): Whether to normalize column names to standard format.
            
        Returns:
            dict: Dictionary mapping ticker symbols to their corresponding DataFrames
        """
        timeframe_dir = os.path.join(basedir, timeframe)
        
        # Check if timeframe directory exists
        if not os.path.exists(timeframe_dir):
            raise ValueError(f"Timeframe directory not found: {timeframe_dir}")
        
        # If tickers not specified, get all CSV files in the directory
        if tickers is None:
            csv_files = glob.glob(os.path.join(timeframe_dir, "*.csv"))
            tickers = [os.path.splitext(os.path.basename(file))[0] for file in csv_files]
        
        result = {}
        for ticker in tickers:
            file_path = os.path.join(timeframe_dir, f"{ticker}.csv")
            if os.path.exists(file_path):
                try:
                    df = DataLoader.load_ohlcv_from_csv(file_path, normalize_columns=normalize_columns)
                    result[ticker] = df
                except Exception as e:
                    print(f"Error loading {ticker}: {str(e)}")
            else:
                print(f"File not found for ticker {ticker}: {file_path}")
        
        return result 