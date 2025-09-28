"""
Simple integration example of sample weights with ML pipeline.
"""

import pandas as pd
import numpy as np
from sample_weights import get_sample_weights_for_ml
from models.random_forest import RandomForestModel


def create_sample_trading_data(n_trades: int = 1000) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create sample trading data for demonstration.
    
    Args:
        n_trades: Number of trades to generate
        
    Returns:
        tuple: (events DataFrame, close prices Series)
    """
    np.random.seed(42)
    
    # Generate dates
    dates = pd.date_range('2020-01-01', periods=n_trades, freq='D')
    
    # Generate close prices (random walk)
    close_prices = pd.Series(100 + np.cumsum(np.random.randn(n_trades) * 0.02), index=dates)
    
    # Generate trade events
    t1_dates = []
    for date in dates:
        days_offset = np.random.randint(1, 10)
        t1_dates.append(date + pd.Timedelta(days=days_offset))
    
    # Generate features and target
    feature1 = np.random.randn(n_trades)
    feature2 = np.random.randn(n_trades)
    feature3 = np.random.randn(n_trades)
    
    # Target based on features
    target = ((feature1 > 0) & (feature2 > 0)) | (feature3 > 1.5)
    target = target.astype(int)
    
    events = pd.DataFrame({
        't1': t1_dates,
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'bin': target
    }, index=dates)
    
    return events, close_prices


def train_with_sample_weights():
    """
    Train model with and without sample weights for comparison.
    """
    print("=== Sample Weights ML Integration Example ===\n")
    
    # 1. Create sample data
    print("1. Creating sample trading data...")
    events, close_prices = create_sample_trading_data(n_trades=1000)
    print(f"   Created {len(events)} trades")
    print(f"   Target distribution: {events['bin'].value_counts().to_dict()}")
    
    # 2. Split data
    train_size = int(0.8 * len(events))
    train_events = events.iloc[:train_size]
    test_events = events.iloc[train_size:]
    train_close = close_prices.iloc[:train_size]
    test_close = close_prices.iloc[train_size:]
    
    print(f"\n2. Split data: {len(train_events)} train, {len(test_events)} test")
    
    # 3. Calculate sample weights
    print("\n3. Calculating sample weights...")
    sample_weights = get_sample_weights_for_ml(
        train_events, 
        train_close, 
        weight_type='w_event',
        t1_col='t1'
    )
    print(f"   Sample weights: min={sample_weights.min():.4f}, max={sample_weights.max():.4f}, mean={sample_weights.mean():.4f}")
    
    # 4. Train models
    print("\n4. Training models...")
    feature_columns = ['feature1', 'feature2', 'feature3']
    
    # Model without sample weights
    print("   Training baseline model (no sample weights)...")
    baseline_model = RandomForestModel(n_estimators=100, random_state=42)
    baseline_model.fit(train_events, feature_columns=feature_columns)
    
    # Model with sample weights
    print("   Training model with sample weights...")
    weighted_model = RandomForestModel(n_estimators=100, random_state=42)
    weighted_model.fit(train_events, feature_columns=feature_columns, sample_weights=sample_weights)
    
    # 5. Evaluate models
    print("\n5. Evaluating models...")
    
    # Baseline model predictions
    baseline_pred = baseline_model.predict(test_events)
    baseline_accuracy = np.mean(baseline_pred == test_events['bin'])
    
    # Weighted model predictions
    weighted_pred = weighted_model.predict(test_events)
    weighted_accuracy = np.mean(weighted_pred == test_events['bin'])
    
    # Calculate weighted accuracy (using sample weights as importance)
    test_weights = get_sample_weights_for_ml(test_events, test_close, weight_type='w_event', t1_col='t1')
    baseline_weighted_acc = np.average(baseline_pred == test_events['bin'], weights=test_weights)
    weighted_weighted_acc = np.average(weighted_pred == test_events['bin'], weights=test_weights)
    
    print("\nModel Performance Results:")
    print(f"{'Metric':<20} {'Baseline':<15} {'With Weights':<15}")
    print("-" * 50)
    print(f"{'Accuracy':<20} {baseline_accuracy:<15.4f} {weighted_accuracy:<15.4f}")
    print(f"{'Weighted Accuracy':<20} {baseline_weighted_acc:<15.4f} {weighted_weighted_acc:<15.4f}")
    
    # 6. Feature importance comparison
    print("\n6. Feature importance comparison:")
    baseline_importance = baseline_model.model.feature_importances_
    weighted_importance = weighted_model.model.feature_importances_
    
    for i, feature in enumerate(feature_columns):
        print(f"   {feature}: {baseline_importance[i]:.4f} (baseline) vs {weighted_importance[i]:.4f} (weighted)")
    
    print("\n=== Example completed ===")
    
    return baseline_model, weighted_model, sample_weights


if __name__ == "__main__":
    train_with_sample_weights() 