"""
Example of ML Performance Integration

This example demonstrates how to:
1. Train ML models on trading data
2. Convert predictions to trading decisions
3. Analyze performance of ML-based strategies
4. Compare different ML approaches
"""

import pandas as pd
import numpy as np
from data_process import DataLoader
from indicators import SimpleSupportResistance
from strategies import SimpleSRStrategy
from ml import (
    MultipleSamplesDataset, 
    MultipleSamplesTrainer, 
    MLPerformanceIntegrator,
    RandomForestModel
)
from ml.feature_engineering import FeatureEngineer


def create_sample_data():
    """Create sample trading data for demonstration."""
    print("Loading sample data...")
    
    # Load crypto data
    symbols = ["BTCUSDT", "ETHUSDT"]
    dfs = DataLoader.load_crypto_data_single_timeframe(
        symbols=symbols,
        timeframe="1h",
        start_date="2023-01-01",
        end_date="2024-01-01",
        data_root="./data/crypto"
    )
    
    # Add indicators
    indicators = [SimpleSupportResistance(lookback=20)]
    for indicator in indicators:
        dfs = indicator.calculate(dfs, append=True)
    
    # Generate events and trades
    strategy = SimpleSRStrategy(params={
        'lookback': 20,
        'hold_periods': 50,
        'barrier_method': 'simple',
        'window': 40,
        'multiplier': [2, 2],
        'min_ret': 0.001
    })
    
    events = strategy.generate_events(dfs, set_barriers=True)
    trades = strategy.generate_trades(dfs)
    
    return dfs, events, trades


def prepare_ml_dataset(dfs, trades):
    """Prepare data for ML training."""
    print("Preparing ML dataset...")
    
    # Create feature engineer
    feature_engineer = FeatureEngineer()
    
    # Prepare samples for ML
    samples = {}
    for symbol in dfs.keys():
        if symbol in trades and len(trades[symbol]) > 0:
            # Get trade data with features
            trade_data = dfs[symbol].loc[trades[symbol].index]
            
            # Add target column (bin from trades)
            trade_data['bin'] = trades[symbol]['bin'].values
            
            # Add entry/exit prices for performance calculation
            trade_data['entry_price'] = trades[symbol]['entry_price'].values
            trade_data['exit_price'] = trades[symbol]['exit_price'].values
            
            samples[symbol] = trade_data
    
    # Create multiple samples dataset
    dataset = MultipleSamplesDataset('symbols')
    
    # Add samples to dataset
    for symbol, data in samples.items():
        dataset.add_sample(symbol, data)
    
    # Split data (simple split for demonstration)
    all_samples = list(samples.keys())
    train_samples = all_samples[:len(all_samples)//2]  # First half for training
    test_samples = all_samples[len(all_samples)//2:]   # Second half for testing
    
    # Create split
    dataset.splits['default'] = {
        'train': train_samples,
        'val': [],
        'test': test_samples
    }
    
    return dataset, train_samples, test_samples


def train_and_evaluate_models(dataset, train_samples, test_samples):
    """Train models and evaluate performance."""
    print("Training and evaluating models...")
    
    # Create trainer
    trainer = MultipleSamplesTrainer(
        model_class=RandomForestModel,
        n_estimators=100,
        random_state=42
    )
    
    # Create performance integrator
    integrator = MLPerformanceIntegrator(
        initial_capital=10000.0,
        risk_free_rate=0.02,
        timeframe="1H"
    )
    
    # Train different approaches
    approaches = ['single_models', 'universal_model', 'ensemble']
    results = {}
    
    for approach in approaches:
        print(f"\n--- Training {approach} ---")
        
        # Evaluate cross-validation performance
        performance = integrator.evaluate_cross_validation_performance(
            trainer=trainer,
            dataset=dataset,
            train_samples=train_samples,
            test_samples=test_samples,
            training_method=approach,
            threshold=0.0,  # Use all non-zero predictions
            min_confidence=0.0
        )
        
        results[approach] = performance
        
        # Print summary
        integrator.print_performance_summary(performance)
    
    return results


def compare_model_performance(results):
    """Compare performance of different approaches."""
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    comparison_data = []
    
    for approach, result in results.items():
        if 'overall_performance' in result:
            overall = result['overall_performance']
            
            row = {
                'Approach': approach,
                'Total Return (%)': overall.get('total_return_mean', 0),
                'Sharpe Ratio': overall.get('sharpe_ratio_mean', 0),
                'Max Drawdown (%)': overall.get('max_drawdown_mean', 0),
                'Win Rate (%)': overall.get('win_rate_mean', 0),
                'Profit Factor': overall.get('profit_factor_mean', 0),
                'Total Trades': overall.get('total_trades_mean', 0)
            }
            comparison_data.append(row)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False, float_format='%.2f'))
    
    # Find best performing model
    if len(comparison_data) > 0:
        best_return_idx = comparison_df['Total Return (%)'].idxmax()
        best_sharpe_idx = comparison_df['Sharpe Ratio'].idxmax()
        
        print(f"\nBest Total Return: {comparison_df.loc[best_return_idx, 'Approach']}")
        print(f"Best Sharpe Ratio: {comparison_df.loc[best_sharpe_idx, 'Approach']}")


def analyze_individual_predictions(dataset, train_samples, test_samples):
    """Analyze individual model predictions."""
    print("\n--- Analyzing Individual Predictions ---")
    
    # Create trainer and integrator
    trainer = MultipleSamplesTrainer(
        model_class=RandomForestModel,
        n_estimators=100,
        random_state=42
    )
    
    integrator = MLPerformanceIntegrator(
        initial_capital=10000.0,
        risk_free_rate=0.02,
        timeframe="1H"
    )
    
    # Train single models
    models = trainer.train_single_models(dataset, train_samples)
    
    # Get predictions
    predictions = trainer.predict_single_models(dataset, test_samples)
    
    # Analyze performance with different thresholds
    thresholds = [0.0, 0.1, 0.2, 0.3]
    threshold_results = {}
    
    for threshold in thresholds:
        print(f"\nAnalyzing with threshold: {threshold}")
        
        performance = integrator.analyze_ml_strategy_performance(
            predictions=predictions,
            dataset=dataset,
            test_samples=test_samples,
            threshold=threshold,
            min_confidence=0.0
        )
        
        threshold_results[threshold] = performance
        
        # Print summary
        integrator.print_performance_summary(performance)
    
    return threshold_results


def main():
    """Main function demonstrating ML performance integration."""
    print("ML Performance Integration Example")
    print("="*50)
    
    # Step 1: Create sample data
    dfs, events, trades = create_sample_data()
    
    # Step 2: Prepare ML dataset
    dataset, train_samples, test_samples = prepare_ml_dataset(dfs, trades)
    
    print(f"Training samples: {train_samples}")
    print(f"Testing samples: {test_samples}")
    
    # Step 3: Train and evaluate different approaches
    results = train_and_evaluate_models(dataset, train_samples, test_samples)
    
    # Step 4: Compare model performance
    compare_model_performance(results)
    
    # Step 5: Analyze individual predictions with different thresholds
    threshold_results = analyze_individual_predictions(dataset, train_samples, test_samples)
    
    # Step 6: Save results
    print("\nSaving results...")
    import json
    
    # Convert results to JSON-serializable format
    def convert_for_json(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_for_json(results)
    
    with open('ml_performance_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print("Results saved to ml_performance_results.json")
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main() 