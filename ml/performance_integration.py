"""
ML Performance Integration Module

This module integrates machine learning predictions with trading performance analysis.
It provides functionality to:
- Convert ML predictions to trading signals
- Analyze performance of ML-based trading strategies
- Evaluate strategies across different time periods and symbols
- Compare different ML models and approaches
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from ..performance import PerformanceAnalyzer, TradesAnalyzer
from .multiple_samples import MultipleSamplesTrainer, MultipleSamplesEvaluator
from .multiple_samples.dataset import MultipleSamplesDataset
from ..models.base_model import BaseModel


class MLPerformanceIntegrator:
    """
    Integrates ML predictions with trading performance analysis.
    
    This class provides methods to:
    - Convert ML predictions to trading decisions
    - Analyze performance of ML-based strategies
    - Compare different ML approaches
    - Generate comprehensive performance reports
    """
    
    def __init__(self, initial_capital: float = 10000.0, 
                 risk_free_rate: float = 0.0,
                 timeframe: str = "1D"):
        """
        Initialize the ML performance integrator.
        
        Args:
            initial_capital: Initial capital for performance calculations
            risk_free_rate: Risk-free rate for Sharpe ratio calculations
            timeframe: Timeframe for performance calculations
        """
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.timeframe = timeframe
        self.performance_results = {}
        
    def convert_predictions_to_trades(self, 
                                    predictions: Dict[str, np.ndarray],
                                    dataset: MultipleSamplesDataset,
                                    test_samples: List[str],
                                    threshold: float = 0.0,
                                    min_confidence: float = 0.0) -> Dict[str, pd.DataFrame]:
        """
        Convert ML predictions to trading decisions and generate trade data.
        
        Args:
            predictions: Dictionary of predictions {sample_id: predictions}
            dataset: MultipleSamplesDataset with samples
            test_samples: List of sample IDs
            threshold: Threshold for making trading decisions (0 = use all non-zero predictions)
            min_confidence: Minimum confidence for making trades
            
        Returns:
            Dict[str, pd.DataFrame]: Trade data for each sample
        """
        trades = {}
        
        for sample_id in test_samples:
            if sample_id not in predictions:
                continue
                
            sample_data = dataset.get_sample(sample_id)
            sample_predictions = predictions[sample_id]
            
            # Create trading decisions based on predictions
            trade_decisions = self._create_trading_decisions(
                sample_predictions, threshold, min_confidence
            )
            
            # Generate trade data
            sample_trades = self._generate_trade_data(
                sample_data, trade_decisions, sample_id
            )
            
            trades[sample_id] = sample_trades
        
        return trades
    
    def _create_trading_decisions(self, 
                                 predictions: np.ndarray,
                                 threshold: float = 0.0,
                                 min_confidence: float = 0.0) -> np.ndarray:
        """
        Create trading decisions from ML predictions.
        
        Args:
            predictions: Raw ML predictions
            threshold: Decision threshold
            min_confidence: Minimum confidence threshold
            
        Returns:
            np.ndarray: Trading decisions (1: long, -1: short, 0: no trade)
        """
        decisions = np.zeros_like(predictions)
        
        # Apply threshold-based decisions
        if threshold > 0:
            # Use threshold for binary decisions
            decisions[predictions > threshold] = 1
            decisions[predictions < -threshold] = -1
        else:
            # Use non-zero predictions as signals
            decisions[predictions > 0] = 1
            decisions[predictions < 0] = -1
        
        # Apply confidence filter if specified
        if min_confidence > 0:
            confidence = np.abs(predictions)
            decisions[confidence < min_confidence] = 0
        
        return decisions
    
    def _generate_trade_data(self, 
                            sample_data: pd.DataFrame,
                            decisions: np.ndarray,
                            sample_id: str) -> pd.DataFrame:
        """
        Generate trade data from trading decisions.
        
        Args:
            sample_data: Sample data with features and prices
            decisions: Trading decisions
            sample_id: Sample identifier
            
        Returns:
            pd.DataFrame: Trade data with entry/exit information
        """
        # Find trade entry points
        trade_entries = decisions != 0
        
        if not np.any(trade_entries):
            return pd.DataFrame()
        
        trades_list = []
        
        for i in range(len(sample_data)):
            if trade_entries[i]:
                # Create trade entry
                trade = {
                    'sample_id': sample_id,
                    'entry_time': sample_data.index[i],
                    'entry_price': sample_data['Close'].iloc[i] if 'Close' in sample_data.columns else sample_data.iloc[i, 0],
                    'direction': decisions[i],
                    'bin': decisions[i],  # For compatibility with existing analysis
                    'entry_features': sample_data.iloc[i].to_dict()
                }
                
                # Add exit information if available
                if 'exit_price' in sample_data.columns:
                    trade['exit_price'] = sample_data['exit_price'].iloc[i]
                    trade['exit_time'] = sample_data.index[i]  # Simplified for now
                    
                    # Calculate returns
                    if trade['direction'] == 1:  # Long
                        trade['returns'] = (trade['exit_price'] - trade['entry_price']) / trade['entry_price']
                    else:  # Short
                        trade['returns'] = (trade['entry_price'] - trade['exit_price']) / trade['entry_price']
                
                trades_list.append(trade)
        
        return pd.DataFrame(trades_list)
    
    def analyze_ml_strategy_performance(self,
                                      predictions: Dict[str, np.ndarray],
                                      dataset: MultipleSamplesDataset,
                                      test_samples: List[str],
                                      threshold: float = 0.0,
                                      min_confidence: float = 0.0) -> Dict:
        """
        Analyze performance of ML-based trading strategy.
        
        Args:
            predictions: ML predictions for each sample
            dataset: MultipleSamplesDataset with samples
            test_samples: List of sample IDs to analyze
            threshold: Decision threshold
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dict: Comprehensive performance analysis
        """
        # Convert predictions to trades
        trades = self.convert_predictions_to_trades(
            predictions, dataset, test_samples, threshold, min_confidence
        )
        
        # Analyze performance for each sample
        sample_performance = {}
        overall_performance = {}
        
        for sample_id in test_samples:
            if sample_id not in trades or len(trades[sample_id]) == 0:
                continue
            
            sample_trades = trades[sample_id]
            
            # Create performance analyzer
            analyzer = PerformanceAnalyzer(
                sample_trades,
                initial_capital=self.initial_capital,
                risk_free_rate=self.risk_free_rate,
                timeframe=self.timeframe
            )
            
            # Get performance metrics
            basic_metrics = analyzer.get_basic_metrics()
            detailed_metrics = analyzer.get_detailed_metrics()
            
            sample_performance[sample_id] = {
                'basic_metrics': basic_metrics,
                'detailed_metrics': detailed_metrics,
                'trades_count': len(sample_trades)
            }
        
        # Calculate aggregated performance
        if sample_performance:
            overall_performance = self._calculate_aggregated_performance(sample_performance)
        
        # Create trades analyzer for additional statistics
        trades_analyzer = TradesAnalyzer(trades)
        
        return {
            'sample_performance': sample_performance,
            'overall_performance': overall_performance,
            'trades_analysis': trades_analyzer.get_detailed_stats(),
            'trades_data': trades
        }
    
    def _calculate_aggregated_performance(self, sample_performance: Dict) -> Dict:
        """
        Calculate aggregated performance metrics across all samples.
        
        Args:
            sample_performance: Performance metrics for each sample
            
        Returns:
            Dict: Aggregated performance metrics
        """
        aggregated = {}
        
        # Aggregate basic metrics
        basic_metrics = ['total_trades', 'win_rate', 'profit_factor', 'total_return', 
                        'sharpe_ratio', 'max_drawdown']
        
        for metric in basic_metrics:
            values = [sample['basic_metrics'].get(metric, np.nan) 
                     for sample in sample_performance.values()]
            values = [v for v in values if not pd.isna(v)]
            
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_min'] = np.min(values)
                aggregated[f'{metric}_max'] = np.max(values)
                aggregated[f'{metric}_median'] = np.median(values)
        
        # Aggregate detailed metrics
        detailed_metrics = ['annualized_return', 'volatility', 'avg_win', 'avg_loss']
        
        for metric in detailed_metrics:
            values = [sample['detailed_metrics'].get(metric, np.nan) 
                     for sample in sample_performance.values()]
            values = [v for v in values if not pd.isna(v)]
            
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_min'] = np.min(values)
                aggregated[f'{metric}_max'] = np.max(values)
                aggregated[f'{metric}_median'] = np.median(values)
        
        return aggregated
    
    def compare_ml_models(self,
                         models: Dict[str, BaseModel],
                         dataset: MultipleSamplesDataset,
                         test_samples: List[str],
                         threshold: float = 0.0,
                         min_confidence: float = 0.0) -> Dict:
        """
        Compare performance of different ML models.
        
        Args:
            models: Dictionary of trained models {model_name: model}
            dataset: MultipleSamplesDataset with samples
            test_samples: List of sample IDs to test
            threshold: Decision threshold
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dict: Comparison results
        """
        comparison_results = {}
        
        for model_name, model in models.items():
            # Get predictions from model
            predictions = {}
            for sample_id in test_samples:
                sample_data = dataset.get_sample(sample_id)
                predictions[sample_id] = model.predict(sample_data)
            
            # Analyze performance
            performance = self.analyze_ml_strategy_performance(
                predictions, dataset, test_samples, threshold, min_confidence
            )
            
            comparison_results[model_name] = performance
        
        # Create comparison summary
        comparison_summary = self._create_model_comparison_summary(comparison_results)
        comparison_results['comparison_summary'] = comparison_summary
        
        return comparison_results
    
    def _create_model_comparison_summary(self, comparison_results: Dict) -> pd.DataFrame:
        """
        Create a summary DataFrame comparing different models.
        
        Args:
            comparison_results: Results from compare_ml_models
            
        Returns:
            pd.DataFrame: Comparison summary
        """
        summary_data = []
        
        for model_name, results in comparison_results.items():
            if 'overall_performance' in results:
                row = {'model_name': model_name}
                row.update(results['overall_performance'])
                summary_data.append(row)
        
        if summary_data:
            return pd.DataFrame(summary_data)
        else:
            return pd.DataFrame()
    
    def evaluate_cross_validation_performance(self,
                                            trainer: MultipleSamplesTrainer,
                                            dataset: MultipleSamplesDataset,
                                            train_samples: List[str],
                                            test_samples: List[str],
                                            training_method: str = 'single_models',
                                            threshold: float = 0.0,
                                            min_confidence: float = 0.0) -> Dict:
        """
        Evaluate ML model performance using cross-validation approach.
        
        Args:
            trainer: MultipleSamplesTrainer instance
            dataset: MultipleSamplesDataset with samples
            train_samples: List of samples for training
            test_samples: List of samples for testing
            training_method: Training method ('single_models', 'universal_model', 'ensemble')
            threshold: Decision threshold
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dict: Cross-validation performance results
        """
        # Train models
        if training_method == 'single_models':
            models = trainer.train_single_models(dataset, train_samples)
            predictions = trainer.predict_single_models(dataset, test_samples)
        elif training_method == 'universal_model':
            model = trainer.train_universal_model(dataset, train_samples)
            predictions = trainer.predict_universal_model(dataset, test_samples)
        elif training_method == 'ensemble':
            models = trainer.train_ensemble(dataset, train_samples)
            predictions = trainer.predict_ensemble(dataset, test_samples)
        else:
            raise ValueError(f"Unknown training method: {training_method}")
        
        # Analyze performance
        performance = self.analyze_ml_strategy_performance(
            predictions, dataset, test_samples, threshold, min_confidence
        )
        
        # Add training information
        performance['training_method'] = training_method
        performance['train_samples'] = train_samples
        performance['test_samples'] = test_samples
        performance['model_summary'] = trainer.get_model_summary()
        
        return performance
    
    def print_performance_summary(self, performance_results: Dict):
        """
        Print a formatted summary of ML strategy performance.
        
        Args:
            performance_results: Results from analyze_ml_strategy_performance
        """
        print("=" * 60)
        print("ML STRATEGY PERFORMANCE SUMMARY")
        print("=" * 60)
        
        if 'overall_performance' in performance_results:
            overall = performance_results['overall_performance']
            
            print(f"Total Return (Mean): {overall.get('total_return_mean', 0):.2f}%")
            print(f"Sharpe Ratio (Mean): {overall.get('sharpe_ratio_mean', 0):.2f}")
            print(f"Max Drawdown (Mean): {overall.get('max_drawdown_mean', 0):.2f}%")
            print(f"Win Rate (Mean): {overall.get('win_rate_mean', 0):.2f}%")
            print(f"Profit Factor (Mean): {overall.get('profit_factor_mean', 0):.2f}")
            print(f"Total Trades (Mean): {overall.get('total_trades_mean', 0):.0f}")
        
        if 'sample_performance' in performance_results:
            sample_perf = performance_results['sample_performance']
            print(f"\nNumber of Samples: {len(sample_perf)}")
            
            # Show sample-by-sample breakdown
            for sample_id, perf in sample_perf.items():
                basic = perf['basic_metrics']
                print(f"\n📊 {sample_id}:")
                print(f"  Total Return: {basic.get('total_return', 0):.2f}%")
                print(f"  Sharpe Ratio: {basic.get('sharpe_ratio', 0):.2f}")
                print(f"  Win Rate: {basic.get('win_rate', 0):.2f}%")
                print(f"  Trades: {basic.get('total_trades', 0)}")
        
        print("=" * 60)
    
    def save_performance_results(self, results: Dict, filepath: str):
        """
        Save performance results to file.
        
        Args:
            results: Performance results
            filepath: Path to save results
        """
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            else:
                return obj
        
        converted_results = convert_numpy(results)
        
        with open(filepath, 'w') as f:
            json.dump(converted_results, f, indent=2) 