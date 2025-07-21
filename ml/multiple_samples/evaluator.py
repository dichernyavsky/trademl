"""
Multiple Samples Evaluator for Machine Learning.

This module provides functionality for evaluating ML models on multiple samples
with comprehensive metrics and stability analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
from ..models.base_model import BaseModel
from .dataset import MultipleSamplesDataset
from ...performance import calculate_returns, calculate_sharpe_ratio, calculate_max_drawdown


class MultipleSamplesEvaluator:
    """
    Evaluates models on multiple samples with comprehensive metrics.
    
    This class provides methods for:
    - Individual sample evaluation
    - Aggregated metrics across samples
    - Stability analysis
    - Model comparison
    """
    
    def __init__(self):
        """Initialize the multiple samples evaluator."""
        self.evaluation_results = {}
        
    def evaluate_model(self, model: BaseModel, dataset: MultipleSamplesDataset, 
                      test_samples: List[str], target_column: str = 'bin') -> Dict:
        """
        Evaluate a model on test samples.
        
        Args:
            model: Trained model to evaluate
            dataset: MultipleSamplesDataset with samples
            test_samples: List of sample IDs to evaluate on
            target_column: Name of the target column
            
        Returns:
            Dict: Evaluation results
        """
        results = {
            'individual_results': {},
            'aggregated_results': {},
            'stability_metrics': {}
        }
        
        # Evaluate on each sample individually
        for sample_id in test_samples:
            sample_data = dataset.get_sample(sample_id)
            
            # Make predictions
            predictions = model.predict(sample_data)
            
            # Calculate metrics for this sample
            sample_results = self._calculate_sample_metrics(
                sample_data, predictions, target_column
            )
            
            results['individual_results'][sample_id] = sample_results
        
        # Calculate aggregated metrics
        results['aggregated_results'] = self._calculate_aggregated_metrics(
            results['individual_results']
        )
        
        # Calculate stability metrics
        results['stability_metrics'] = self._calculate_stability_metrics(
            results['individual_results']
        )
        
        return results
    
    def evaluate_multiple_models(self, models: Dict[str, BaseModel], 
                               dataset: MultipleSamplesDataset, 
                               test_samples: List[str], 
                               target_column: str = 'bin') -> Dict:
        """
        Evaluate multiple models and compare them.
        
        Args:
            models: Dictionary of trained models {model_name: model}
            dataset: MultipleSamplesDataset with samples
            test_samples: List of sample IDs to evaluate on
            target_column: Name of the target column
            
        Returns:
            Dict: Comparison results
        """
        comparison_results = {}
        
        # Evaluate each model
        for model_name, model in models.items():
            model_results = self.evaluate_model(
                model, dataset, test_samples, target_column
            )
            comparison_results[model_name] = model_results
        
        # Create comparison summary
        comparison_summary = self._create_comparison_summary(comparison_results)
        comparison_results['comparison_summary'] = comparison_summary
        
        return comparison_results
    
    def _calculate_sample_metrics(self, sample_data: pd.DataFrame, 
                                 predictions: np.ndarray, 
                                 target_column: str) -> Dict:
        """
        Calculate metrics for a single sample.
        
        Args:
            sample_data: Sample data with features and target
            predictions: Model predictions
            target_column: Name of the target column
            
        Returns:
            Dict: Sample metrics
        """
        # Get true labels
        y_true = sample_data[target_column].values
        
        # Classification metrics
        accuracy = np.mean(predictions == y_true)
        
        # Calculate precision, recall, F1 for each class
        precision_1 = np.mean(y_true[predictions == 1] == 1) if np.sum(predictions == 1) > 0 else 0
        recall_1 = np.mean(predictions[y_true == 1] == 1) if np.sum(y_true == 1) > 0 else 0
        f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0
        
        precision_neg1 = np.mean(y_true[predictions == -1] == -1) if np.sum(predictions == -1) > 0 else 0
        recall_neg1 = np.mean(predictions[y_true == -1] == -1) if np.sum(y_true == -1) > 0 else 0
        f1_neg1 = 2 * (precision_neg1 * recall_neg1) / (precision_neg1 + recall_neg1) if (precision_neg1 + recall_neg1) > 0 else 0
        
        # Trading metrics (if available)
        trading_metrics = {}
        if 'entry_price' in sample_data.columns and 'exit_price' in sample_data.columns:
            # Calculate returns based on predictions
            entry_prices = sample_data['entry_price'].values
            exit_prices = sample_data['exit_price'].values
            
            # Calculate returns for predicted trades (predictions != 0)
            trade_mask = predictions != 0
            if np.sum(trade_mask) > 0:
                trade_returns = (exit_prices[trade_mask] - entry_prices[trade_mask]) / entry_prices[trade_mask]
                trade_returns *= predictions[trade_mask]  # Apply direction
                
                trading_metrics = {
                    'total_return': np.sum(trade_returns),
                    'mean_return': np.mean(trade_returns),
                    'return_std': np.std(trade_returns),
                    'sharpe_ratio': calculate_sharpe_ratio(trade_returns),
                    'max_drawdown': calculate_max_drawdown(trade_returns),
                    'win_rate': np.mean(trade_returns > 0),
                    'num_trades': len(trade_returns)
                }
        
        return {
            'accuracy': accuracy,
            'precision_1': precision_1,
            'recall_1': recall_1,
            'f1_1': f1_1,
            'precision_neg1': precision_neg1,
            'recall_neg1': recall_neg1,
            'f1_neg1': f1_neg1,
            'trading_metrics': trading_metrics,
            'num_samples': len(sample_data)
        }
    
    def _calculate_aggregated_metrics(self, individual_results: Dict) -> Dict:
        """
        Calculate aggregated metrics across all samples.
        
        Args:
            individual_results: Dictionary of individual sample results
            
        Returns:
            Dict: Aggregated metrics
        """
        # Extract metrics for aggregation
        metrics_to_aggregate = [
            'accuracy', 'precision_1', 'recall_1', 'f1_1',
            'precision_neg1', 'recall_neg1', 'f1_neg1'
        ]
        
        aggregated = {}
        
        for metric in metrics_to_aggregate:
            values = [results[metric] for results in individual_results.values()]
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_min'] = np.min(values)
            aggregated[f'{metric}_max'] = np.max(values)
            aggregated[f'{metric}_median'] = np.median(values)
        
        # Aggregate trading metrics if available
        trading_metrics = []
        for results in individual_results.values():
            if results['trading_metrics']:
                trading_metrics.append(results['trading_metrics'])
        
        if trading_metrics:
            trading_aggregated = {}
            for metric in ['total_return', 'mean_return', 'return_std', 'sharpe_ratio', 
                          'max_drawdown', 'win_rate']:
                values = [tm[metric] for tm in trading_metrics if metric in tm]
                if values:
                    trading_aggregated[f'{metric}_mean'] = np.mean(values)
                    trading_aggregated[f'{metric}_std'] = np.std(values)
                    trading_aggregated[f'{metric}_min'] = np.min(values)
                    trading_aggregated[f'{metric}_max'] = np.max(values)
                    trading_aggregated[f'{metric}_median'] = np.median(values)
            
            aggregated['trading_metrics'] = trading_aggregated
        
        return aggregated
    
    def _calculate_stability_metrics(self, individual_results: Dict) -> Dict:
        """
        Calculate stability metrics across samples.
        
        Args:
            individual_results: Dictionary of individual sample results
            
        Returns:
            Dict: Stability metrics
        """
        stability = {}
        
        # Calculate coefficient of variation for key metrics
        key_metrics = ['accuracy', 'f1_1', 'f1_neg1']
        
        for metric in key_metrics:
            values = [results[metric] for results in individual_results.values()]
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if mean_val != 0:
                cv = std_val / abs(mean_val)  # Coefficient of variation
                stability[f'{metric}_cv'] = cv
                stability[f'{metric}_stability_score'] = 1 / (1 + cv)  # Higher is more stable
        
        # Calculate rank correlation between samples (if we have multiple metrics)
        if len(individual_results) > 1:
            # Create a matrix of metrics across samples
            sample_names = list(individual_results.keys())
            metric_names = ['accuracy', 'f1_1', 'f1_neg1']
            
            metric_matrix = []
            for metric in metric_names:
                row = [individual_results[sample][metric] for sample in sample_names]
                metric_matrix.append(row)
            
            metric_matrix = np.array(metric_matrix)
            
            # Calculate rank correlation between metrics
            from scipy.stats import spearmanr
            correlations = []
            for i in range(len(metric_names)):
                for j in range(i+1, len(metric_names)):
                    corr, _ = spearmanr(metric_matrix[i], metric_matrix[j])
                    correlations.append(corr)
            
            if correlations:
                stability['metric_rank_correlation_mean'] = np.mean(correlations)
                stability['metric_rank_correlation_std'] = np.std(correlations)
        
        return stability
    
    def _create_comparison_summary(self, comparison_results: Dict) -> pd.DataFrame:
        """
        Create a summary DataFrame comparing multiple models.
        
        Args:
            comparison_results: Results from evaluate_multiple_models
            
        Returns:
            pd.DataFrame: Comparison summary
        """
        summary_data = []
        
        for model_name, results in comparison_results.items():
            if 'aggregated_results' in results:
                row = {'model_name': model_name}
                row.update(results['aggregated_results'])
                summary_data.append(row)
        
        if summary_data:
            return pd.DataFrame(summary_data)
        else:
            return pd.DataFrame()
    
    def get_best_model(self, comparison_results: Dict, 
                      metric: str = 'accuracy_mean') -> str:
        """
        Get the best performing model based on a metric.
        
        Args:
            comparison_results: Results from evaluate_multiple_models
            metric: Metric to use for comparison
            
        Returns:
            str: Name of the best model
        """
        if 'comparison_summary' not in comparison_results:
            raise ValueError("No comparison summary available")
        
        summary_df = comparison_results['comparison_summary']
        
        if metric not in summary_df.columns:
            raise ValueError(f"Metric {metric} not found in comparison results")
        
        best_idx = summary_df[metric].idxmax()
        return summary_df.loc[best_idx, 'model_name']
    
    def plot_comparison(self, comparison_results: Dict, 
                       metrics: List[str] = None) -> None:
        """
        Plot comparison of models (placeholder for visualization).
        
        Args:
            comparison_results: Results from evaluate_multiple_models
            metrics: List of metrics to plot
        """
        # TODO: Implement visualization
        # This would create bar charts, box plots, etc. comparing models
        print("Plotting functionality not yet implemented")
    
    def save_results(self, results: Dict, filepath: str) -> None:
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results
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
            else:
                return obj
        
        converted_results = convert_numpy(results)
        
        with open(filepath, 'w') as f:
            json.dump(converted_results, f, indent=2) 