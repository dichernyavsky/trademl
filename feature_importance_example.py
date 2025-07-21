"""
Feature Importance Analysis Example for Jupyter Notebook

This example shows how to use the Feature Importance module to analyze
which indicators are most important for your trading strategy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import our modules
from ml.feature_importance import FeatureImportanceAnalyzer
from ml.feature_engineering import FeatureEngineer
from indicators import ADX, Stochastic, SimpleSupportResistance, WilliamsR
from strategies import SimpleSRStrategy
from data_process import CryptoDataLoader

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data(symbol='BTCUSDT', timeframe='1_hour', limit=2000):
    """
    Load and prepare data for analysis.
    
    Args:
        symbol: Trading symbol
        timeframe: Data timeframe
        limit: Number of data points
        
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Loading {symbol} data ({timeframe}, {limit} points)...")
    
    loader = CryptoDataLoader()
    data = loader.load_data(symbol, timeframe, limit=limit)
    
    print(f"✅ Loaded data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    return data

def generate_trades_with_features(data, strategy_params=None):
    """
    Generate trades and add technical indicators as features.
    
    Args:
        data: OHLCV DataFrame
        strategy_params: Strategy parameters
        
    Returns:
        DataFrame with trades and features
    """
    if strategy_params is None:
        strategy_params = {'lookback': 20, 'threshold': 0.02}
    
    print("Generating trades...")
    
    # Generate trades using strategy
    strategy = SimpleSRStrategy(**strategy_params)
    trades = strategy.generate_trades(data)
    
    print(f"✅ Generated {len(trades)} trades")
    
    # Create feature engineer
    feature_engineer = FeatureEngineer()
    
    # Add various technical indicators
    print("Adding technical indicators...")
    
    # Trend indicators
    feature_engineer.add_indicator(ADX(period=14))
    feature_engineer.add_indicator(ADX(period=21))
    
    # Momentum indicators
    feature_engineer.add_indicator(Stochastic(period=14))
    feature_engineer.add_indicator(Stochastic(period=21))
    feature_engineer.add_indicator(WilliamsR(period=14))
    
    # Support/Resistance
    feature_engineer.add_indicator(SimpleSupportResistance(lookback=20))
    feature_engineer.add_indicator(SimpleSupportResistance(lookback=50))
    
    # Generate features
    trades_with_features = feature_engineer.generate_features(data, trades)
    
    print(f"✅ Trades with features shape: {trades_with_features.shape}")
    
    # Show feature columns
    feature_columns = [col for col in trades_with_features.columns 
                      if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 
                                   'bin', 'direction', 'entry_time', 'exit_time']]
    print(f"Feature columns: {feature_columns}")
    
    return trades_with_features

def analyze_feature_importance(trades_df, methods=['mdi', 'mda'], cv_splits=5):
    """
    Analyze feature importance using multiple methods.
    
    Args:
        trades_df: DataFrame with trades and features
        methods: List of importance methods
        cv_splits: Number of CV splits
        
    Returns:
        FeatureImportanceAnalyzer object and importance DataFrame
    """
    print(f"Analyzing feature importance using methods: {methods}")
    
    # Create analyzer
    analyzer = FeatureImportanceAnalyzer(
        base_estimator='rf',
        methods=methods,
        cv_splits=cv_splits,
        embargo_pct=0.01,
        random_state=42
    )
    
    # Analyze trades
    importance_df = analyzer.analyze_trades(trades_df)
    
    print(f"✅ Analysis completed! Shape: {importance_df.shape}")
    
    return analyzer, importance_df

def display_results(analyzer, importance_df, n_top=10):
    """
    Display and visualize feature importance results.
    
    Args:
        analyzer: FeatureImportanceAnalyzer object
        importance_df: Importance DataFrame
        n_top: Number of top features to display
    """
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE RESULTS")
    print("="*60)
    
    # 1. Top features by mean importance
    print(f"\n📊 Top {n_top} Features (Mean Importance):")
    top_features = analyzer.get_top_features(n=n_top, method='mean')
    print(top_features)
    
    # 2. Method comparison
    print(f"\n📈 Method Comparison (Top {n_top}):")
    comparison = analyzer.compare_methods(n=n_top)
    print(comparison)
    
    # 3. Filter important features
    important_features = analyzer.filter_features(
        trades_df=None,  # Already analyzed
        threshold=0.01,
        method='mean'
    )
    print(f"\n🎯 Important Features (threshold=0.01): {important_features}")
    
    # 4. Visualizations
    create_visualizations(analyzer, importance_df, n_top)

def create_visualizations(analyzer, importance_df, n_top=10):
    """
    Create visualizations of feature importance.
    
    Args:
        analyzer: FeatureImportanceAnalyzer object
        importance_df: Importance DataFrame
        n_top: Number of top features to plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Top features bar plot
    top_features = analyzer.get_top_features(n=n_top, method='mean')
    axes[0, 0].barh(range(len(top_features)), top_features.values)
    axes[0, 0].set_yticks(range(len(top_features)))
    axes[0, 0].set_yticklabels(top_features.index)
    axes[0, 0].set_title(f'Top {n_top} Features (Mean Importance)')
    axes[0, 0].set_xlabel('Importance Score')
    
    # 2. Method comparison heatmap
    comparison = analyzer.compare_methods(n=n_top)
    sns.heatmap(comparison.T, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=axes[0, 1])
    axes[0, 1].set_title('Method Comparison Heatmap')
    
    # 3. Method correlation
    correlation = importance_df.corr()
    sns.heatmap(correlation, annot=True, fmt='.3f', cmap='coolwarm', ax=axes[1, 0])
    axes[1, 0].set_title('Method Correlation Matrix')
    
    # 4. Feature importance distribution
    importance_df.mean(axis=1).hist(bins=20, ax=axes[1, 1], alpha=0.7)
    axes[1, 1].axvline(0.01, color='red', linestyle='--', label='Threshold (0.01)')
    axes[1, 1].set_title('Feature Importance Distribution')
    axes[1, 1].set_xlabel('Mean Importance Score')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

def save_results(analyzer, filepath='feature_importance_results.csv'):
    """
    Save feature importance results to file.
    
    Args:
        analyzer: FeatureImportanceAnalyzer object
        filepath: Path to save results
    """
    try:
        analyzer.save_results(filepath, format='csv')
        print(f"✅ Results saved to {filepath}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

def main_example():
    """
    Complete example workflow.
    """
    print("🚀 Feature Importance Analysis Example")
    print("="*50)
    
    # 1. Load data
    data = load_and_prepare_data('BTCUSDT', '1_hour', 2000)
    
    # 2. Generate trades with features
    trades_with_features = generate_trades_with_features(data)
    
    # 3. Analyze feature importance
    analyzer, importance_df = analyze_feature_importance(
        trades_with_features, 
        methods=['mdi', 'mda', 'sfi'],
        cv_splits=5
    )
    
    # 4. Display results
    display_results(analyzer, importance_df, n_top=10)
    
    # 5. Save results
    save_results(analyzer)
    
    print("\n🎉 Analysis completed successfully!")
    
    return analyzer, importance_df

# Example usage for Jupyter notebook
if __name__ == "__main__":
    # Run the complete example
    analyzer, results = main_example()
    
    # You can also run individual functions:
    # data = load_and_prepare_data('ETHUSDT', '1_hour', 1000)
    # trades = generate_trades_with_features(data)
    # analyzer, results = analyze_feature_importance(trades)
    # display_results(analyzer, results) 