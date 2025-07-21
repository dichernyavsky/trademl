"""
Trading Performance Analysis Module

This module provides tools for analyzing trading performance including:
- Basic metrics (returns, Sharpe ratio, drawdown)
- Trade analysis
- Risk metrics
- Performance visualization
"""

from .analyzer import PerformanceAnalyzer
from .metrics import calculate_returns, calculate_sharpe_ratio, calculate_max_drawdown, get_periods_per_year
from .trades_analyzer import TradesAnalyzer, quick_trades_analysis

__all__ = [
    'PerformanceAnalyzer',
    'calculate_returns',
    'calculate_sharpe_ratio',
    'calculate_max_drawdown',
    'get_periods_per_year',
    'TradesAnalyzer',
    'quick_trades_analysis'
] 