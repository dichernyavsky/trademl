#!/usr/bin/env python3
"""
Test script to demonstrate the new entry_price_mode functionality
in SimpleSRStrategy.
"""

import pandas as pd
import numpy as np
from strategies.simple_sr_strategy import SimpleSRStrategy

# Create sample data with clear support/resistance levels
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=100, freq='1H')
data = pd.DataFrame({
    'Open': 100 + np.random.randn(100) * 0.5,
    'High': 100 + np.random.randn(100) * 0.5 + 1,
    'Low': 100 + np.random.randn(100) * 0.5 - 1,
    'Close': 100 + np.random.randn(100) * 0.5,
    'Volume': np.random.randint(1000, 10000, 100)
}, index=dates)

# Add some clear support/resistance levels
data.loc[20:25, 'High'] = 102.5  # Resistance level
data.loc[20:25, 'Close'] = 102.0
data.loc[30:35, 'Low'] = 98.0   # Support level  
data.loc[30:35, 'Close'] = 98.5

# Test with close price mode (default)
print("=== Testing with entry_price_mode='close' ===")
strategy_close = SimpleSRStrategy({
    'lookback': 10,
    'entry_offset': 1,
    'entry_price_mode': 'close'
})

events_close = strategy_close._generate_raw_events(data)
if not events_close.empty:
    print("Events found with close price mode:")
    print(events_close[['direction', 'entry_price']].head())
    print(f"Entry prices are close prices: {events_close['entry_price'].iloc[0] == data.loc[events_close.index[0], 'Close']}")
else:
    print("No events found with close price mode")

print("\n=== Testing with entry_price_mode='breakout' ===")
strategy_breakout = SimpleSRStrategy({
    'lookback': 10,
    'entry_offset': 1,
    'entry_price_mode': 'breakout'
})

events_breakout = strategy_breakout._generate_raw_events(data)
if not events_breakout.empty:
    print("Events found with breakout price mode:")
    print(events_breakout[['direction', 'entry_price']].head())
    print(f"Entry prices are breakout prices (not close): {events_breakout['entry_price'].iloc[0] != data.loc[events_breakout.index[0], 'Close']}")
else:
    print("No events found with breakout price mode")

print("\n=== Testing parameter validation ===")
try:
    strategy_invalid = SimpleSRStrategy({'entry_price_mode': 'invalid'})
    print("ERROR: Should have raised ValueError for invalid entry_price_mode")
except ValueError as e:
    print(f"✓ Correctly caught invalid parameter: {e}")

print("\n=== Testing backward compatibility ===")
strategy_default = SimpleSRStrategy()  # No params
print(f"✓ Default entry_price_mode: {strategy_default.entry_price_mode}")
