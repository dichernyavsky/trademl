"""
Simple test for sample weights functionality.
"""

import pandas as pd
import numpy as np
from sample_weights import (
    build_event_weights,
    get_sample_weights_for_ml
)


def test_sample_weights():
    """Test basic sample weights functionality."""
    print("Testing Sample Weights...")
    
    # Create test data
    n_samples = 100
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Create close prices
    close_prices = pd.Series(100 + np.cumsum(np.random.randn(n_samples) * 0.02), index=dates)
    
    # Create events (trades)
    t1_dates = []
    for date in dates:
        days_offset = np.random.randint(1, 10)
        t1_dates.append(date + pd.Timedelta(days=days_offset))
    
    events = pd.DataFrame({
        't1': t1_dates,
        'bin': np.random.randint(0, 2, n_samples)
    }, index=dates)
    
    print(f"Created test data with {len(events)} events")
    
    # Test build_event_weights
    print("\n1. Testing build_event_weights...")
    weights_df = build_event_weights(close_prices, events, t1_col='t1')
    print(f"   Weight columns: {list(weights_df.columns)}")
    print(f"   w_event stats: min={weights_df['w_event'].min():.4f}, max={weights_df['w_event'].max():.4f}, mean={weights_df['w_event'].mean():.4f}")
    
    # Test get_sample_weights_for_ml
    print("\n2. Testing get_sample_weights_for_ml...")
    sample_weights = get_sample_weights_for_ml(events, close_prices, weight_type='w_event')
    print(f"   Sample weights: min={sample_weights.min():.4f}, max={sample_weights.max():.4f}, mean={sample_weights.mean():.4f}")
    
    # Test different weight types
    print("\n3. Testing different weight types...")
    for weight_type in ['w_event', 'w_return', 'w_uniqueness', 'w_time']:
        weights = get_sample_weights_for_ml(events, close_prices, weight_type=weight_type)
        print(f"   {weight_type}: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_sample_weights() 