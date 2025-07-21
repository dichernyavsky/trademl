# Advanced Technical Indicators

This document describes the advanced technical indicators that have been implemented in the TradeML framework.

## Overview

The advanced indicators module (`indicators/advanced.py`) contains sophisticated technical indicators that combine multiple basic indicators or provide specialized calculations for enhanced market analysis.

## Implemented Indicators

### 1. BreakoutGapATR

**Formula:** `(Close - SR) / ATR`

**Description:** Measures the distance of price from support/resistance levels normalized by volatility (ATR). This helps identify potential breakout opportunities and overbought/oversold conditions.

**Parameters:**
- `sr_lookback` (int): Lookback period for support/resistance calculation (default: 20)
- `atr_window` (int): Window size for ATR calculation (default: 14)
- `column` (str): Column to use for calculations (default: 'Close')

**Output Columns:**
- `BreakoutGapATR_{column}_{sr_lookback}_{atr_window}_Support`
- `BreakoutGapATR_{column}_{sr_lookback}_{atr_window}_Resistance`

**Interpretation:**
- Values > 2: Price significantly above support (bullish)
- Values < -2: Price significantly below resistance (bearish)
- Values near 0: Price at support/resistance levels

### 2. VWAPDistance

**Formula:** `(Close - VWAP) / VWAP`

**Description:** Measures the percentage distance of current price from the Volume Weighted Average Price (VWAP). Useful for identifying fair value and potential mean reversion opportunities.

**Parameters:**
- `column` (str): Column to use for calculations (default: 'Close')

**Output Columns:**
- `VWAPDistance_{column}`

**Interpretation:**
- Values > 0.05: Price significantly above VWAP (potentially overbought)
- Values < -0.05: Price significantly below VWAP (potentially oversold)
- Values near 0: Price at fair value

### 3. BollingerBandwidth

**Formula:** `(Upper - Lower) / Middle`

**Description:** Measures the relative width of Bollinger Bands, indicating volatility levels. Wide bands suggest high volatility, narrow bands suggest low volatility.

**Parameters:**
- `window` (int): Window size for Bollinger Bands calculation (default: 20)
- `num_std` (float): Number of standard deviations for the bands (default: 2.0)
- `column` (str): Column to use for calculations (default: 'Close')

**Output Columns:**
- `BollingerBandwidth_{column}_{window}_std_{num_std}`

**Interpretation:**
- Values > 0.2: High volatility (bands are wide)
- Values < 0.1: Low volatility (bands are narrow)
- Values 0.1-0.2: Moderate volatility

### 4. NormalizedMomentum

**Description:** Combines RSI, Stochastic, and CCI indicators with normalization to specific ranges for easier comparison and combination.

**Formulas:**
- RSI: `RSI / 100` → [0, 1]
- Stochastic: `Stoch / 100` → [0, 1]
- CCI: `CCI / 400` → [-1, 1]

**Parameters:**
- `rsi_window` (int): Window size for RSI calculation (default: 14)
- `stoch_window` (int): Window size for Stochastic calculation (default: 14)
- `cci_window` (int): Window size for CCI calculation (default: 20)
- `column` (str): Column to use for calculations (default: 'Close')

**Output Columns:**
- `NormalizedRSI_{column}_{rsi_window}`
- `NormalizedStoch_{column}_{stoch_window}`
- `NormalizedCCI_{column}_{cci_window}`

**Interpretation:**
- RSI/Stoch near 1: Overbought conditions
- RSI/Stoch near 0: Oversold conditions
- CCI > 0.5: Strong bullish momentum
- CCI < -0.5: Strong bearish momentum

### 5. PercentRank

**Formula:** Percentile rank of current price within a rolling window

**Description:** Calculates the percentile rank of the current close price within a rolling window of N periods. Useful for identifying relative price position and potential reversal points.

**Parameters:**
- `window` (int): Window size for percentile calculation (default: 20)
- `column` (str): Column to use for calculations (default: 'Close')

**Output Columns:**
- `PercentRank_{column}_{window}`

**Interpretation:**
- Values > 0.8: Price in top 20% of recent range (overbought)
- Values < 0.2: Price in bottom 20% of recent range (oversold)
- Values 0.2-0.8: Price in middle range

## Usage Examples

### Basic Usage

```python
from indicators.advanced import (
    BreakoutGapATR, 
    VWAPDistance, 
    BollingerBandwidth, 
    NormalizedMomentum, 
    PercentRank
)

# Initialize indicators
breakout_atr = BreakoutGapATR(sr_lookback=20, atr_window=14)
vwap_dist = VWAPDistance()
bb_bandwidth = BollingerBandwidth(window=20, num_std=2.0)
norm_momentum = NormalizedMomentum(rsi_window=14, stoch_window=14, cci_window=20)
pct_rank = PercentRank(window=20)

# Calculate indicators
data = breakout_atr.calculate(data, append=True)
data = vwap_dist.calculate(data, append=True)
data = bb_bandwidth.calculate(data, append=True)
data = norm_momentum.calculate(data, append=True)
data = pct_rank.calculate(data, append=True)
```

### Signal Generation

```python
def generate_signals(data):
    """Generate trading signals based on indicator combinations."""
    
    # Get latest values
    support_gap = data['BreakoutGapATR_Close_20_14_Support'].iloc[-1]
    resistance_gap = data['BreakoutGapATR_Close_20_14_Resistance'].iloc[-1]
    vwap_dist = data['VWAPDistance_Close'].iloc[-1]
    rsi_norm = data['NormalizedRSI_Close_14'].iloc[-1]
    pct_rank = data['PercentRank_Close_20'].iloc[-1]
    
    # Signal logic
    bullish_signals = 0
    bearish_signals = 0
    
    # Breakout signals
    if support_gap > 1.5:
        bullish_signals += 1
    if resistance_gap < -1.5:
        bearish_signals += 1
    
    # VWAP signals
    if vwap_dist > 0.03:
        bullish_signals += 1
    elif vwap_dist < -0.03:
        bearish_signals += 1
    
    # Momentum signals
    if rsi_norm > 0.7:
        bullish_signals += 1
    elif rsi_norm < 0.3:
        bearish_signals += 1
    
    # Percent rank signals
    if pct_rank < 0.3:
        bullish_signals += 1
    elif pct_rank > 0.7:
        bearish_signals += 1
    
    # Generate final signal
    if bullish_signals >= 2 and bearish_signals == 0:
        return "STRONG BUY"
    elif bearish_signals >= 2 and bullish_signals == 0:
        return "STRONG SELL"
    elif bullish_signals > bearish_signals:
        return "WEAK BUY"
    elif bearish_signals > bullish_signals:
        return "WEAK SELL"
    else:
        return "NEUTRAL"
```

## Integration with Feature Engineering

These indicators can be easily integrated with the existing feature engineering framework:

```python
from ml.feature_engineering.feature_engineer import FeatureEngineer
from indicators.advanced import BreakoutGapATR, VWAPDistance

# Create feature engineer
feature_engineer = FeatureEngineer()

# Add advanced indicators
feature_engineer.add_indicator(BreakoutGapATR(sr_lookback=20, atr_window=14))
feature_engineer.add_indicator(VWAPDistance())
feature_engineer.add_indicator(BollingerBandwidth(window=20, num_std=2.0))
feature_engineer.add_indicator(NormalizedMomentum())
feature_engineer.add_indicator(PercentRank(window=20))

# Generate features
features = feature_engineer.generate_features(data)
```

## Testing

Run the test script to verify all indicators work correctly:

```bash
python test_advanced_indicators.py
```

Run the example script to see practical usage:

```bash
python example_advanced_indicators.py
```

## Notes

- All indicators follow the `BaseIndicator` interface for consistency
- Indicators handle missing data gracefully with NaN values
- Column names are automatically generated based on parameters
- All calculations include small epsilon values (1e-9) to prevent division by zero
- The indicators are designed to work with OHLCV data format

## Dependencies

- pandas
- numpy
- Existing indicator modules (support_resistance, volatility, etc.) 