"""
TA-Lib Indicators Module

This module provides wrapper classes for TA-Lib indicators organized by category.
All indicators are compatible with the BaseIndicator interface and can be used
for machine learning applications with proper normalization.
"""

# Import all indicator classes
from .oscillators import RSI, Stochastic, WilliamsR, CCI, ROC, Momentum, MACD, ADX, Aroon, TRIX
from .volatility import ATR, BollingerBands, StandardDeviation
from .volume import OBV, AccumulationDistribution, ChaikinMoneyFlow, MFI

__all__ = [
    # Oscillators
    'RSI', 'Stochastic', 'WilliamsR', 'CCI', 'ROC', 'Momentum',
    
    # Normalized indicators
    'MACD', 'ADX', 'Aroon', 'TRIX', 'MFI',
    
    # Volatility indicators
    'ATR', 'BollingerBands', 'StandardDeviation',
    
    # Volume indicators
    'OBV', 'AccumulationDistribution', 'ChaikinMoneyFlow'
] 