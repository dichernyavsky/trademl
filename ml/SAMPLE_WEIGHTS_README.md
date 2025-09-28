# Sample Weights Module

## Overview

The Sample Weights module provides functionality to calculate sample weights for machine learning models in trading applications. Sample weights allow you to give different importance to different training samples based on various criteria.

This module implements the sample weighting approach described in the discussion, focusing on:
- **Event uniqueness**: How unique each trading event is
- **Return-based weights**: Weights based on returns during the event period
- **Time decay**: Weights that decay over time
- **Composite event weights**: Combination of all factors

## Quick Start

### Basic Usage

```python
from ml.sample_weights import get_sample_weights_for_ml

# Calculate sample weights for ML training
sample_weights = get_sample_weights_for_ml(
    events=events_df,           # DataFrame with trading events
    close=close_prices,         # Series with close prices
    weight_type='w_event',      # Type of weight to use
    t1_col='t1'                 # Column name for exit times
)

# Use with ML model
model.fit(trades_data, sample_weights=sample_weights)
```

### Complete Example

```python
import pandas as pd
import numpy as np
from ml.sample_weights import get_sample_weights_for_ml
from ml.models.random_forest import RandomForestModel

# Create sample data
events = pd.DataFrame({
    't1': exit_times,           # Exit times for each trade
    'feature1': feature1,
    'feature2': feature2,
    'bin': target
}, index=entry_times)

close_prices = pd.Series(prices, index=timestamps)

# Calculate sample weights
sample_weights = get_sample_weights_for_ml(
    events, close_prices, 
    weight_type='w_event'
)

# Train model with sample weights
model = RandomForestModel(n_estimators=100)
model.fit(events, sample_weights=sample_weights)
```

## Functions

### `get_sample_weights_for_ml()`

Main function to calculate sample weights for ML training.

```python
def get_sample_weights_for_ml(events: pd.DataFrame, 
                             close: pd.Series,
                             weight_type: str = 'w_event',
                             t1_col: str = 't1',
                             **kwargs) -> np.ndarray:
```

**Parameters:**
- `events`: DataFrame with trading events (must have index as entry times)
- `close`: Series with close prices (must have same index as events)
- `weight_type`: Type of weight to use
  - `'w_event'`: Composite event weight (default)
  - `'w_return'`: Return-based weight
  - `'w_uniqueness'`: Uniqueness weight
  - `'w_time'`: Time decay weight
- `t1_col`: Column name containing exit times
- `**kwargs`: Additional parameters passed to `build_event_weights()`

**Returns:**
- `np.ndarray`: Sample weights for ML training

### `build_event_weights()`

Calculate all types of weights for events.

```python
def build_event_weights(close: pd.Series,
                        events: pd.DataFrame,
                        t1_col: str = 't1',
                        fallback_freq: str = '1min',
                        clf_last_w: float = 1.0,
                        use_abs_ret: bool = True,
                        normalize_ret: bool = True) -> pd.DataFrame:
```

**Returns:**
- DataFrame with columns: `['w_uniqueness', 'w_return', 'w_time', 'w_event']`

## Weight Types

### 1. Event Uniqueness (`w_uniqueness`)
Measures how unique each trading event is based on overlapping events.

### 2. Return-based Weights (`w_return`)
Weights based on returns during the event period, normalized by the number of overlapping events.

### 3. Time Decay (`w_time`)
Weights that decay over time based on cumulative uniqueness.

### 4. Composite Event Weight (`w_event`)
Combination of all three weights: `w_event = w_uniqueness * w_return * w_time`

## Integration with ML Pipeline

### With BaseModel

All models inheriting from `BaseModel` support sample weights:

```python
from ml.models.random_forest import RandomForestModel

# Calculate weights
sample_weights = get_sample_weights_for_ml(events, close_prices)

# Train model
model = RandomForestModel(n_estimators=100)
model.fit(trades_data, sample_weights=sample_weights)
```

### With Multiple Samples Trainer

The `MultipleSamplesTrainer` has been updated to support sample weights:

```python
from ml.multiple_samples import MultipleSamplesTrainer
from ml.sample_weights import get_sample_weights_for_ml

# Set up trainer
trainer = MultipleSamplesTrainer(RandomForestModel, n_estimators=100)

# Train with sample weights
models = trainer.train_single_models(
    dataset=dataset,
    train_samples=train_samples,
    sample_weights_manager=weight_manager,
    weight_calculator_name='w_event'
)
```

## Data Requirements

### Events DataFrame
- **Index**: Entry times (DatetimeIndex)
- **Required columns**:
  - `t1`: Exit times for each event
  - `bin`: Target variable (0/1 for classification)
  - Feature columns for ML training

### Close Prices Series
- **Index**: Timestamps matching the events
- **Values**: Close prices at each timestamp

## Examples

### Example 1: Basic Usage

```python
# Calculate weights
weights = get_sample_weights_for_ml(events, close_prices)

# Train model
model.fit(events, sample_weights=weights)
```

### Example 2: Different Weight Types

```python
# Try different weight types
for weight_type in ['w_event', 'w_return', 'w_uniqueness', 'w_time']:
    weights = get_sample_weights_for_ml(events, close_prices, weight_type=weight_type)
    model.fit(events, sample_weights=weights)
    # Evaluate model...
```

### Example 3: Custom Parameters

```python
# Use custom parameters
weights = get_sample_weights_for_ml(
    events, close_prices,
    weight_type='w_event',
    clf_last_w=0.5,        # Time decay parameter
    use_abs_ret=False,     # Use raw returns instead of absolute
    normalize_ret=False    # Don't normalize return weights
)
```

## Best Practices

### 1. Data Quality
- Ensure events DataFrame has proper DatetimeIndex
- Verify t1 column contains valid exit times
- Check that close prices cover the entire event period

### 2. Weight Validation
```python
# Check weight distribution
weights = get_sample_weights_for_ml(events, close_prices)
print(f"Weight stats: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")

# Check for extreme values
if np.any(weights > 10):
    print("Warning: Very high weights detected")
```

### 3. Model Comparison
Always compare models with and without sample weights:
```python
# Baseline model
baseline_model.fit(events)

# Weighted model
weighted_model.fit(events, sample_weights=weights)

# Compare performance
baseline_acc = baseline_model.score(test_events)
weighted_acc = weighted_model.score(test_events)
```

## Performance Considerations

- **Memory**: Large event datasets can consume significant memory
- **Computation**: Weight calculation involves time series operations
- **Vectorization**: Functions use vectorized operations for efficiency

## Troubleshooting

### Common Issues

1. **Missing t1 column**: Ensure events DataFrame has the specified t1 column
2. **Index mismatch**: Verify events and close prices have compatible indices
3. **Invalid dates**: Check that entry and exit times are valid datetime objects

### Debugging

```python
# Check data structure
print(f"Events shape: {events.shape}")
print(f"Events columns: {list(events.columns)}")
print(f"Events index type: {type(events.index)}")
print(f"Close prices shape: {close_prices.shape}")

# Test weight calculation
try:
    weights = get_sample_weights_for_ml(events, close_prices)
    print(f"Weights calculated successfully: {len(weights)} weights")
except Exception as e:
    print(f"Error calculating weights: {e}")
``` 