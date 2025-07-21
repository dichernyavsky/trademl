"""
Basic trading performance metrics
"""

import numpy as np
import pandas as pd
from typing import Union, Optional


def get_periods_per_year(timeframe: str) -> int:
    """
    Calculate number of periods per year based on timeframe.
    
    Args:
        timeframe: Timeframe string (e.g., '1H', '1D', '1M', '5M', '1h', '1d', etc.)
        
    Returns:
        Number of periods per year
    """
    timeframe = timeframe.upper()
    
    # Extract number and unit
    if len(timeframe) < 2:
        raise ValueError(f"Invalid timeframe format: {timeframe}")
    
    # Handle cases like '1H', '5M', '1D', '1W'
    if timeframe[-1] in ['H', 'M', 'D', 'W']:
        number = int(timeframe[:-1]) if len(timeframe) > 1 else 1
        unit = timeframe[-1]
    else:
        # Handle cases like '1HOUR', '5MIN', '1DAY', '1WEEK'
        if 'HOUR' in timeframe or 'H' in timeframe:
            number = int(''.join(filter(str.isdigit, timeframe)))
            unit = 'H'
        elif 'MIN' in timeframe or 'M' in timeframe:
            number = int(''.join(filter(str.isdigit, timeframe)))
            unit = 'M'
        elif 'DAY' in timeframe or 'D' in timeframe:
            number = int(''.join(filter(str.isdigit, timeframe)))
            unit = 'D'
        elif 'WEEK' in timeframe or 'W' in timeframe:
            number = int(''.join(filter(str.isdigit, timeframe)))
            unit = 'W'
        else:
            raise ValueError(f"Unknown timeframe unit in: {timeframe}")
    
    # Calculate periods per year
    if unit == 'M':  # Minutes
        periods_per_year = (365 * 24 * 60) // number
    elif unit == 'H':  # Hours
        periods_per_year = (365 * 24) // number
    elif unit == 'D':  # Days
        periods_per_year = 365 // number
    else:
        raise ValueError(f"Unknown timeframe unit: {unit}")
    
    return periods_per_year


def calculate_returns(trades_df: pd.DataFrame, 
                     entry_price_col: str = 'entry_price',
                     exit_price_col: str = 'exit_price',
                     direction_col: str = 'direction') -> pd.Series:
    """
    Calculate returns for each trade.
    
    Args:
        trades_df: DataFrame with trade information
        entry_price_col: Column name for entry prices
        exit_price_col: Column name for exit prices  
        direction_col: Column name for trade direction (1 for long, -1 for short)
        
    Returns:
        Series with returns for each trade
    """
    entry_prices = trades_df[entry_price_col]
    exit_prices = trades_df[exit_price_col]
    directions = trades_df[direction_col]
    
    # Calculate returns based on direction
    # For longs: (exit - entry) / entry
    # For shorts: (entry - exit) / entry
    returns = directions * (exit_prices - entry_prices) / entry_prices
    
    return returns


def calculate_sharpe_ratio(returns: Union[pd.Series, np.ndarray], 
                          risk_free_rate: float = 0.0,
                          timeframe: str = "1D") -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Series or array of returns
        risk_free_rate: Risk-free rate (annualized)
        timeframe: Timeframe string (e.g., '1H', '1D', '1M', '5M', '1h', '1d', etc.)
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return np.nan
    
    # Remove NaN values
    if isinstance(returns, pd.Series):
        returns_clean = returns.dropna()
    else:
        returns_clean = returns[~np.isnan(returns)]
    
    if len(returns_clean) == 0:
        return np.nan
    
    # Calculate periods per year from timeframe
    periods_per_year = get_periods_per_year(timeframe)
    
    # Calculate mean and std
    mean_return = np.mean(returns_clean)
    std_return = np.std(returns_clean, ddof=1)  # Sample standard deviation
    
    if std_return == 0:
        return np.nan
    
    # Annualize
    annualized_return = mean_return * periods_per_year
    annualized_volatility = std_return * np.sqrt(periods_per_year)
    
    # Calculate Sharpe ratio
    sharpe = (annualized_return - risk_free_rate) / annualized_volatility
    
    return sharpe


def calculate_max_drawdown(equity_curve: Union[pd.Series, np.ndarray]) -> dict:
    """
    Calculate maximum drawdown and related metrics.
    
    Args:
        equity_curve: Series or array of cumulative equity values
        
    Returns:
        Dictionary with max drawdown, start and end indices
    """
    if len(equity_curve) == 0:
        return {
            'max_drawdown': np.nan,
            'max_drawdown_pct': np.nan,
            'start_idx': np.nan,
            'end_idx': np.nan
        }
    
    # Convert to numpy array if needed
    if isinstance(equity_curve, pd.Series):
        equity_array = equity_curve.values
        index = equity_curve.index
    else:
        equity_array = equity_curve
        index = np.arange(len(equity_array))
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(equity_array)
    
    # Calculate drawdown
    drawdown = (equity_array - running_max) / running_max
    
    # Find maximum drawdown
    max_dd_idx = np.argmin(drawdown)
    max_dd = drawdown[max_dd_idx]
    
    # Find peak before max drawdown
    peak_idx = np.argmax(equity_array[:max_dd_idx + 1])
    
    return {
        'max_drawdown': max_dd,
        'max_drawdown_pct': abs(max_dd) * 100,
        'start_idx': index[peak_idx] if hasattr(index, '__getitem__') else peak_idx,
        'end_idx': index[max_dd_idx] if hasattr(index, '__getitem__') else max_dd_idx
    }


def calculate_win_rate(trades_df: pd.DataFrame, 
                      returns_col: str = 'returns') -> float:
    """
    Calculate win rate (percentage of profitable trades).
    
    Args:
        trades_df: DataFrame with trade information
        returns_col: Column name for returns
        
    Returns:
        Win rate as percentage
    """
    if returns_col not in trades_df.columns:
        # Calculate returns if not present
        returns = calculate_returns(trades_df)
    else:
        returns = trades_df[returns_col]
    
    # Remove NaN values
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return np.nan
    
    # Calculate win rate
    winning_trades = returns_clean > 0
    win_rate = np.mean(winning_trades) * 100
    
    return win_rate


def calculate_profit_factor(trades_df: pd.DataFrame,
                           returns_col: str = 'returns') -> float:
    """
    Calculate profit factor (gross profit / gross loss).
    
    Args:
        trades_df: DataFrame with trade information
        returns_col: Column name for returns
        
    Returns:
        Profit factor
    """
    if returns_col not in trades_df.columns:
        # Calculate returns if not present
        returns = calculate_returns(trades_df)
    else:
        returns = trades_df[returns_col]
    
    # Remove NaN values
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return np.nan
    
    # Calculate gross profit and loss
    gross_profit = returns_clean[returns_clean > 0].sum()
    gross_loss = abs(returns_clean[returns_clean < 0].sum())
    
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else np.nan
    
    profit_factor = gross_profit / gross_loss
    
    return profit_factor


def calculate_avg_trade_duration(trades_df: pd.DataFrame,
                                entry_time_col: str = 'entry_time',
                                exit_time_col: str = 'exit_time') -> pd.Timedelta:
    """
    Calculate average trade duration.
    
    Args:
        trades_df: DataFrame with trade information
        entry_time_col: Column name for entry times
        exit_time_col: Column name for exit times
        
    Returns:
        Average trade duration as Timedelta
    """
    if entry_time_col not in trades_df.columns or exit_time_col not in trades_df.columns:
        return pd.NaT
    
    # Calculate durations
    durations = trades_df[exit_time_col] - trades_df[entry_time_col]
    
    # Remove invalid durations
    valid_durations = durations[durations.notna() & (durations > pd.Timedelta(0))]
    
    if len(valid_durations) == 0:
        return pd.NaT
    
    avg_duration = valid_durations.mean()
    
    return avg_duration 