"""
Multiple Samples Trainer for Machine Learning.

This module provides functionality for training ML models on multiple samples
with different strategies (single model per sample, universal model, ensemble).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Type, Any
from ..models.base_model import BaseModel
from .dataset import MultipleSamplesDataset


class MultipleSamplesTrainer:
    """
    Trains models on multiple samples with different strategies.
    
    This class supports various training approaches:
    - Single model per sample
    - Universal model (one model for all samples)
    - Ensemble approach (multiple models, ensemble predictions)
    """
    
    def __init__(self, model_class: Type[BaseModel], feature_engineer=None, **model_params):
        """
        Initialize the multiple samples trainer.
        
        Args:
            model_class: Class of the model to train (e.g., RandomForestModel)
            feature_engineer: Optional feature engineer for feature generation
            **model_params: Parameters to pass to the model constructor
        """
        self.model_class = model_class
        self.feature_engineer = feature_engineer
        self.model_params = model_params
        self.models = {}  # {sample_id: trained_model}
        self.training_results = {}  # {sample_id: training_results}
        self.universal_model = None  # Single model trained on all samples
        
    def train_single_models(self, dataset: MultipleSamplesDataset, 
                           train_samples: List[str], 
                           sample_weights_manager=None,
                           weight_calculator_name=None,
                           **kwargs) -> Dict[str, BaseModel]:
        """
        Train separate model for each sample.
        
        Args:
            dataset: MultipleSamplesDataset with samples
            train_samples: List of sample IDs to train on
            sample_weights_manager: Optional SampleWeightsManager for calculating weights
            weight_calculator_name: Name of the weight calculator to use
            **kwargs: Additional training parameters
            
        Returns:
            Dict[str, BaseModel]: Dictionary of trained models
        """
        self.models = {}
        
        for sample_id in train_samples:
            print(f"Training model for sample: {sample_id}")
            
            # Get sample data
            sample_data = dataset.get_sample(sample_id)
            
            # Generate features if feature engineer is provided
            if self.feature_engineer:
                sample_data = self.feature_engineer.generate_features(
                    {sample_id: sample_data}, sample_data
                )
            
            # Calculate sample weights if manager is provided
            sample_weights = None
            if sample_weights_manager and weight_calculator_name:
                try:
                    sample_weights = sample_weights_manager.calculate_weights(
                        weight_calculator_name, sample_data
                    )
                    print(f"   Using {weight_calculator_name} weights: min={sample_weights.min():.4f}, max={sample_weights.max():.4f}")
                except Exception as e:
                    print(f"   Warning: Could not calculate weights for {sample_id}: {e}")
            
            # Create and train model
            model = self.model_class(**self.model_params)
            model.fit(sample_data, sample_weights=sample_weights, **kwargs)
            
            self.models[sample_id] = model
            
        return self.models
    
    def train_universal_model(self, dataset: MultipleSamplesDataset, 
                             train_samples: List[str], 
                             sample_weights_manager=None,
                             weight_calculator_name=None,
                             **kwargs) -> BaseModel:
        """
        Train a single model on all samples combined.
        
        Args:
            dataset: MultipleSamplesDataset with samples
            train_samples: List of sample IDs to train on
            sample_weights_manager: Optional SampleWeightsManager for calculating weights
            weight_calculator_name: Name of the weight calculator to use
            **kwargs: Additional training parameters
            
        Returns:
            BaseModel: Trained universal model
        """
        print("Training universal model on all samples")
        
        # Get combined data from all samples
        combined_data = dataset.get_combined_split_data(
            split_name='default', split_type='train'
        )
        
        # Generate features if feature engineer is provided
        if self.feature_engineer:
            # Create a dict structure for feature engineer
            sample_data_dict = {}
            for sample_id in train_samples:
                sample_data = dataset.get_sample(sample_id)
                sample_data_dict[sample_id] = sample_data
            
            # Generate features for each sample
            enriched_samples = {}
            for sample_id, sample_data in sample_data_dict.items():
                enriched_sample = self.feature_engineer.generate_features(
                    {sample_id: sample_data}, sample_data
                )
                enriched_sample['sample_id'] = sample_id
                enriched_samples[sample_id] = enriched_sample
            
            # Combine enriched samples
            combined_data = pd.concat(enriched_samples.values(), ignore_index=True)
        
        # Calculate sample weights if manager is provided
        sample_weights = None
        if sample_weights_manager and weight_calculator_name:
            try:
                sample_weights = sample_weights_manager.calculate_weights(
                    weight_calculator_name, combined_data
                )
                print(f"Using {weight_calculator_name} weights: min={sample_weights.min():.4f}, max={sample_weights.max():.4f}")
            except Exception as e:
                print(f"Warning: Could not calculate weights for universal model: {e}")
        
        # Create and train universal model
        self.universal_model = self.model_class(**self.model_params)
        self.universal_model.fit(combined_data, sample_weights=sample_weights, **kwargs)
        
        return self.universal_model
    
    def train_ensemble(self, dataset: MultipleSamplesDataset, 
                      train_samples: List[str], ensemble_method: str = 'voting', **kwargs) -> Dict[str, BaseModel]:
        """
        Train ensemble of models.
        
        Args:
            dataset: MultipleSamplesDataset with samples
            train_samples: List of sample IDs to train on
            ensemble_method: Method for ensemble ('voting', 'stacking', 'bagging')
            **kwargs: Additional training parameters
            
        Returns:
            Dict[str, BaseModel]: Dictionary of trained models
        """
        if ensemble_method == 'voting':
            return self._train_voting_ensemble(dataset, train_samples, **kwargs)
        elif ensemble_method == 'stacking':
            return self._train_stacking_ensemble(dataset, train_samples, **kwargs)
        elif ensemble_method == 'bagging':
            return self._train_bagging_ensemble(dataset, train_samples, **kwargs)
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")
    
    def _train_voting_ensemble(self, dataset: MultipleSamplesDataset, 
                              train_samples: List[str], **kwargs) -> Dict[str, BaseModel]:
        """
        Train voting ensemble (separate models, average predictions).
        """
        return self.train_single_models(dataset, train_samples, **kwargs)
    
    def _train_stacking_ensemble(self, dataset: MultipleSamplesDataset, 
                                train_samples: List[str], **kwargs) -> Dict[str, BaseModel]:
        """
        Train stacking ensemble (meta-learner on base model predictions).
        """
        # First train base models
        base_models = self.train_single_models(dataset, train_samples, **kwargs)
        
        # TODO: Implement meta-learner training
        # This would involve:
        # 1. Getting predictions from base models on validation set
        # 2. Training a meta-learner on these predictions
        # 3. Storing both base models and meta-learner
        
        return base_models
    
    def _train_bagging_ensemble(self, dataset: MultipleSamplesDataset, 
                               train_samples: List[str], **kwargs) -> Dict[str, BaseModel]:
        """
        Train bagging ensemble (bootstrap samples of the same data).
        """
        # For bagging, we create multiple models on bootstrap samples
        # of the combined data
        combined_data = dataset.get_combined_split_data(
            split_name='default', split_type='train'
        )
        
        n_models = kwargs.get('n_models', 5)
        self.models = {}
        
        for i in range(n_models):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(
                len(combined_data), 
                size=len(combined_data), 
                replace=True
            )
            bootstrap_data = combined_data.iloc[bootstrap_indices]
            
            # Train model on bootstrap sample
            model = self.model_class(**self.model_params)
            model.fit(bootstrap_data, **kwargs)
            
            self.models[f'bagging_model_{i}'] = model
        
        return self.models
    
    def predict_single_models(self, dataset: MultipleSamplesDataset, 
                            test_samples: List[str]) -> Dict[str, np.ndarray]:
        """
        Make predictions using single models approach.
        
        Args:
            dataset: MultipleSamplesDataset with samples
            test_samples: List of sample IDs to predict on
            
        Returns:
            Dict[str, np.ndarray]: Predictions for each sample
        """
        predictions = {}
        
        for sample_id in test_samples:
            if sample_id not in self.models:
                raise ValueError(f"No trained model found for sample {sample_id}")
            
            # Get sample data
            sample_data = dataset.get_sample(sample_id)
            
            # Generate features if feature engineer is provided
            if self.feature_engineer:
                sample_data = self.feature_engineer.generate_features(
                    {sample_id: sample_data}, sample_data
                )
            
            # Make predictions
            predictions[sample_id] = self.models[sample_id].predict(sample_data)
        
        return predictions
    
    def predict_universal_model(self, dataset: MultipleSamplesDataset, 
                               test_samples: List[str]) -> Dict[str, np.ndarray]:
        """
        Make predictions using universal model.
        
        Args:
            dataset: MultipleSamplesDataset with samples
            test_samples: List of sample IDs to predict on
            
        Returns:
            Dict[str, np.ndarray]: Predictions for each sample
        """
        if self.universal_model is None:
            raise ValueError("Universal model not trained")
        
        predictions = {}
        
        for sample_id in test_samples:
            # Get sample data
            sample_data = dataset.get_sample(sample_id)
            
            # Generate features if feature engineer is provided
            if self.feature_engineer:
                sample_data = self.feature_engineer.generate_features(
                    {sample_id: sample_data}, sample_data
                )
            
            # Make predictions
            predictions[sample_id] = self.universal_model.predict(sample_data)
        
        return predictions
    
    def predict_ensemble(self, dataset: MultipleSamplesDataset, 
                        test_samples: List[str], ensemble_method: str = 'voting') -> Dict[str, np.ndarray]:
        """
        Make ensemble predictions.
        
        Args:
            dataset: MultipleSamplesDataset with samples
            test_samples: List of sample IDs to predict on
            ensemble_method: Method for ensemble ('voting', 'stacking', 'bagging')
            
        Returns:
            Dict[str, np.ndarray]: Ensemble predictions for each sample
        """
        if ensemble_method == 'voting':
            return self._predict_voting_ensemble(dataset, test_samples)
        elif ensemble_method == 'stacking':
            return self._predict_stacking_ensemble(dataset, test_samples)
        elif ensemble_method == 'bagging':
            return self._predict_bagging_ensemble(dataset, test_samples)
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")
    
    def _predict_voting_ensemble(self, dataset: MultipleSamplesDataset, 
                                test_samples: List[str]) -> Dict[str, np.ndarray]:
        """
        Make voting ensemble predictions (average of individual model predictions).
        """
        # Get predictions from all models
        all_predictions = {}
        for sample_id in test_samples:
            sample_predictions = []
            
            for model_id, model in self.models.items():
                if sample_id in model_id or 'bagging' in model_id:
                    # For bagging models or models that can predict any sample
                    sample_data = dataset.get_sample(sample_id)
                    
                    if self.feature_engineer:
                        sample_data = self.feature_engineer.generate_features(
                            {sample_id: sample_data}, sample_data
                        )
                    
                    pred = model.predict(sample_data)
                    sample_predictions.append(pred)
            
            if sample_predictions:
                # Average predictions
                all_predictions[sample_id] = np.mean(sample_predictions, axis=0)
        
        return all_predictions
    
    def _predict_stacking_ensemble(self, dataset: MultipleSamplesDataset, 
                                  test_samples: List[str]) -> Dict[str, np.ndarray]:
        """
        Make stacking ensemble predictions.
        """
        # TODO: Implement stacking predictions
        # This would involve:
        # 1. Getting base model predictions
        # 2. Using meta-learner to combine predictions
        raise NotImplementedError("Stacking ensemble predictions not yet implemented")
    
    def _predict_bagging_ensemble(self, dataset: MultipleSamplesDataset, 
                                 test_samples: List[str]) -> Dict[str, np.ndarray]:
        """
        Make bagging ensemble predictions (average of bootstrap models).
        """
        return self._predict_voting_ensemble(dataset, test_samples)
    
    def get_model_summary(self) -> Dict:
        """
        Get summary of trained models.
        
        Returns:
            Dict: Summary of models and training results
        """
        summary = {
            'total_models': len(self.models),
            'model_ids': list(self.models.keys()),
            'has_universal_model': self.universal_model is not None,
            'model_class': self.model_class.__name__
        }
        
        # Add model-specific information
        model_info = {}
        for model_id, model in self.models.items():
            model_info[model_id] = {
                'is_fitted': model.is_fitted,
                'feature_columns': model.feature_columns,
                'model_type': model.model_type
            }
        summary['model_info'] = model_info
        
        return summary 