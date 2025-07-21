"""
Main performance analyzer class
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union
from .metrics import (
    calculate_returns, calculate_sharpe_ratio, calculate_max_drawdown,
    calculate_win_rate, calculate_profit_factor, calculate_avg_trade_duration,
    get_periods_per_year
)


class PerformanceAnalyzer:
    """
    Main class for analyzing trading performance.
    """
    
    def __init__(self, trades_df: pd.DataFrame, 
                 initial_capital: float = 10000.0,
                 risk_free_rate: float = 0.0,
                 timeframe: str = "1D"):
        """
        Initialize performance analyzer.
        
        Args:
            trades_df: DataFrame with trade information
            initial_capital: Initial capital for equity curve calculation
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            timeframe: Timeframe string (e.g., '1H', '1D', '1M', '5M', '1h', '1d', etc.)
        """
        self.trades_df = trades_df.copy()
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.timeframe = timeframe
        
        # Calculate periods per year from timeframe
        self.periods_per_year = get_periods_per_year(timeframe)
        
        # Calculate returns if not present
        if 'returns' not in self.trades_df.columns:
            self.trades_df['returns'] = calculate_returns(self.trades_df)
        
        # Calculate equity curve
        self.equity_curve = self._calculate_equity_curve()
        
    def _calculate_equity_curve(self) -> pd.Series:
        """
        Calculate equity curve from trades.
        
        Returns:
            Series with cumulative equity values
        """
        if len(self.trades_df) == 0:
            return pd.Series([self.initial_capital])
        
        # Get returns and remove NaN
        returns = self.trades_df['returns'].dropna()
        
        if len(returns) == 0:
            return pd.Series([self.initial_capital])
        
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Calculate equity curve
        equity_curve = self.initial_capital * cumulative_returns
        
        return equity_curve
    
    def get_basic_metrics(self) -> Dict[str, float]:
        """
        Calculate basic performance metrics.
        
        Returns:
            Dictionary with basic metrics
        """
        if len(self.trades_df) == 0:
            return {
                'total_trades': 0,
                'win_rate': np.nan,
                'profit_factor': np.nan,
                'total_return': np.nan,
                'sharpe_ratio': np.nan,
                'max_drawdown': np.nan,
                'avg_trade_duration': np.nan
            }
        
        # Basic trade statistics
        total_trades = len(self.trades_df)
        win_rate = calculate_win_rate(self.trades_df)
        profit_factor = calculate_profit_factor(self.trades_df)
        
        # Return metrics
        total_return = (self.equity_curve.iloc[-1] / self.initial_capital - 1) * 100
        
        # Risk metrics
        returns = self.trades_df['returns'].dropna()
        sharpe_ratio = calculate_sharpe_ratio(returns, self.risk_free_rate, self.timeframe)
        
        # Drawdown
        drawdown_info = calculate_max_drawdown(self.equity_curve)
        max_drawdown = drawdown_info['max_drawdown_pct']
        
        # Duration
        avg_duration = calculate_avg_trade_duration(self.trades_df)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_trade_duration': avg_duration
        }
    
    def get_detailed_metrics(self) -> Dict[str, Union[float, pd.Series]]:
        """
        Calculate detailed performance metrics.
        
        Returns:
            Dictionary with detailed metrics
        """
        basic_metrics = self.get_basic_metrics()
        
        if len(self.trades_df) == 0:
            return basic_metrics
        
        # Additional metrics
        returns = self.trades_df['returns'].dropna()
        
        # Return statistics
        annualized_return = returns.mean() * self.periods_per_year * 100
        volatility = returns.std() * np.sqrt(self.periods_per_year) * 100
        
        # Trade statistics
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        
        avg_win = winning_trades.mean() * 100 if len(winning_trades) > 0 else 0
        avg_loss = losing_trades.mean() * 100 if len(losing_trades) > 0 else 0
        max_win = winning_trades.max() * 100 if len(winning_trades) > 0 else 0
        max_loss = losing_trades.min() * 100 if len(losing_trades) > 0 else 0
        
        # Consecutive wins/losses
        consecutive_wins = self._calculate_consecutive_wins()
        consecutive_losses = self._calculate_consecutive_losses()
        
        detailed_metrics = {
            **basic_metrics,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_win': max_win,
            'max_loss': max_loss,
            'consecutive_wins': consecutive_wins,
            'consecutive_losses': consecutive_losses,
            'equity_curve': self.equity_curve,
            'periods_per_year': self.periods_per_year
        }
        
        return detailed_metrics
    
    def _calculate_consecutive_wins(self) -> int:
        """Calculate maximum consecutive wins."""
        returns = self.trades_df['returns'].dropna()
        if len(returns) == 0:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for ret in returns:
            if ret > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_consecutive_losses(self) -> int:
        """Calculate maximum consecutive losses."""
        returns = self.trades_df['returns'].dropna()
        if len(returns) == 0:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for ret in returns:
            if ret < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def print_summary(self):
        """Print a formatted summary of performance metrics."""
        metrics = self.get_detailed_metrics()
        
        print("=" * 50)
        print("TRADING PERFORMANCE SUMMARY")
        print("=" * 50)
        
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Annualized Return: {metrics['annualized_return']:.2f}%")
        print(f"Volatility: {metrics['volatility']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Average Win: {metrics['avg_win']:.2f}%")
        print(f"Average Loss: {metrics['avg_loss']:.2f}%")
        print(f"Max Win: {metrics['max_win']:.2f}%")
        print(f"Max Loss: {metrics['max_loss']:.2f}%")
        print(f"Consecutive Wins: {metrics['consecutive_wins']}")
        print(f"Consecutive Losses: {metrics['consecutive_losses']}")
        print(f"Periods per Year: {metrics['periods_per_year']}")
        
        if pd.notna(metrics['avg_trade_duration']):
            print(f"Average Trade Duration: {metrics['avg_trade_duration']}")
        
        print("=" * 50)
    
    def get_equity_curve(self) -> pd.Series:
        """Get the equity curve."""
        return self.equity_curve
    
    def get_trades_summary(self) -> pd.DataFrame:
        """Get a summary of trades."""
        if len(self.trades_df) == 0:
            return pd.DataFrame()
        
        summary = self.trades_df.copy()
        
        # Add cumulative return
        summary['cumulative_return'] = (1 + summary['returns']).cumprod()
        
        # Add running equity
        summary['equity'] = self.initial_capital * summary['cumulative_return']
        
        return summary 