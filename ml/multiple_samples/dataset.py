"""
Multiple Samples Dataset for Machine Learning.

This module provides functionality for managing multiple samples (symbols, time periods, timeframes)
for training and evaluating ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from sklearn.model_selection import train_test_split


class MultipleSamplesDataset:
    """
    Manages multiple samples for machine learning.
    
    This class handles multiple samples (symbols, time periods, timeframes)
    and provides methods for splitting, accessing, and managing the data.
    """
    
    def __init__(self, data_structure: str = 'symbols'):
        """
        Initialize the multiple samples dataset.
        
        Args:
            data_structure: Type of data structure ('symbols', 'time_periods', 'timeframes')
        """
        self.data_structure = data_structure
        self.samples = {}  # {sample_id: DataFrameTrades}
        self.metadata = {}  # {sample_id: metadata}
        self.splits = {}  # {split_name: {train: [], val: [], test: []}}
        
    def add_sample(self, sample_id: str, trades_df: pd.DataFrame, 
                  metadata: Optional[Dict] = None) -> None:
        """
        Add a sample to the dataset.
        
        Args:
            sample_id: Unique identifier for the sample
            trades_df: DataFrameTrades with trading data
            metadata: Optional metadata for the sample
        """
        if sample_id in self.samples:
            raise ValueError(f"Sample {sample_id} already exists")
            
        self.samples[sample_id] = trades_df.copy()
        self.metadata[sample_id] = metadata or {}
        
    def remove_sample(self, sample_id: str) -> None:
        """
        Remove a sample from the dataset.
        
        Args:
            sample_id: ID of the sample to remove
        """
        if sample_id in self.samples:
            del self.samples[sample_id]
            del self.metadata[sample_id]
            
            # Remove from all splits
            for split_name in self.splits:
                for split_type in ['train', 'val', 'test']:
                    if sample_id in self.splits[split_name][split_type]:
                        self.splits[split_name][split_type].remove(sample_id)
        else:
            raise ValueError(f"Sample {sample_id} not found")
            
    def get_sample(self, sample_id: str) -> pd.DataFrame:
        """
        Get a specific sample.
        
        Args:
            sample_id: ID of the sample to retrieve
            
        Returns:
            pd.DataFrame: The sample data
        """
        if sample_id not in self.samples:
            raise ValueError(f"Sample {sample_id} not found")
        return self.samples[sample_id]
        
    def get_all_samples(self) -> Dict[str, pd.DataFrame]:
        """
        Get all samples.
        
        Returns:
            Dict[str, pd.DataFrame]: All samples
        """
        return self.samples.copy()
        
    def get_sample_ids(self) -> List[str]:
        """
        Get list of all sample IDs.
        
        Returns:
            List[str]: List of sample IDs
        """
        return list(self.samples.keys())
        
    def get_metadata(self, sample_id: str) -> Dict:
        """
        Get metadata for a specific sample.
        
        Args:
            sample_id: ID of the sample
            
        Returns:
            Dict: Metadata for the sample
        """
        if sample_id not in self.metadata:
            raise ValueError(f"Sample {sample_id} not found")
        return self.metadata[sample_id].copy()
        
    def split_samples(self, split_method: str = 'random', split_name: str = 'default',
                     train_ratio: float = 0.7, val_ratio: float = 0.15, 
                     test_ratio: float = 0.15, random_state: int = 42, **kwargs) -> None:
        """
        Split samples into train/validation/test sets.
        
        Args:
            split_method: Method for splitting ('random', 'time', 'symbol')
            split_name: Name for this split configuration
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters for specific split methods
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
            
        sample_ids = self.get_sample_ids()
        
        if split_method == 'random':
            train_ids, temp_ids = train_test_split(
                sample_ids, 
                train_size=train_ratio, 
                random_state=random_state
            )
            
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            val_ids, test_ids = train_test_split(
                temp_ids, 
                train_size=val_ratio_adjusted, 
                random_state=random_state
            )
            
        elif split_method == 'time':
            # Sort by time period (assuming metadata contains time info)
            sorted_ids = self._sort_samples_by_time(sample_ids)
            n_samples = len(sorted_ids)
            
            train_end = int(n_samples * train_ratio)
            val_end = int(n_samples * (train_ratio + val_ratio))
            
            train_ids = sorted_ids[:train_end]
            val_ids = sorted_ids[train_end:val_end]
            test_ids = sorted_ids[val_end:]
            
        elif split_method == 'symbol':
            # For symbol-based splits, use random but ensure diversity
            train_ids, temp_ids = train_test_split(
                sample_ids, 
                train_size=train_ratio, 
                random_state=random_state,
                stratify=self._get_symbol_groups(sample_ids)
            )
            
            val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
            val_ids, test_ids = train_test_split(
                temp_ids, 
                train_size=val_ratio_adjusted, 
                random_state=random_state
            )
            
        else:
            raise ValueError(f"Unknown split method: {split_method}")
            
        self.splits[split_name] = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids
        }
        
    def get_split(self, split_name: str = 'default', split_type: str = 'train') -> List[str]:
        """
        Get sample IDs for a specific split.
        
        Args:
            split_name: Name of the split configuration
            split_type: Type of split ('train', 'val', 'test')
            
        Returns:
            List[str]: List of sample IDs for the specified split
        """
        if split_name not in self.splits:
            raise ValueError(f"Split {split_name} not found")
        if split_type not in ['train', 'val', 'test']:
            raise ValueError(f"Split type must be 'train', 'val', or 'test'")
            
        return self.splits[split_name][split_type].copy()
        
    def get_split_data(self, split_name: str = 'default', split_type: str = 'train') -> Dict[str, pd.DataFrame]:
        """
        Get sample data for a specific split.
        
        Args:
            split_name: Name of the split configuration
            split_type: Type of split ('train', 'val', 'test')
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary of sample data
        """
        sample_ids = self.get_split(split_name, split_type)
        return {sample_id: self.get_sample(sample_id) for sample_id in sample_ids}
        
    def get_combined_split_data(self, split_name: str = 'default', split_type: str = 'train') -> pd.DataFrame:
        """
        Get combined data for a specific split (all samples concatenated).
        
        Args:
            split_name: Name of the split configuration
            split_type: Type of split ('train', 'val', 'test')
            
        Returns:
            pd.DataFrame: Combined data from all samples in the split
        """
        split_data = self.get_split_data(split_name, split_type)
        
        if not split_data:
            return pd.DataFrame()
            
        # Add sample_id column to each DataFrame
        combined_data = []
        for sample_id, df in split_data.items():
            df_copy = df.copy()
            df_copy['sample_id'] = sample_id
            combined_data.append(df_copy)
            
        return pd.concat(combined_data, ignore_index=True)
        
    def _sort_samples_by_time(self, sample_ids: List[str]) -> List[str]:
        """
        Sort samples by time period (helper method).
        
        Args:
            sample_ids: List of sample IDs to sort
            
        Returns:
            List[str]: Sorted sample IDs
        """
        # This is a placeholder implementation
        # In practice, you would extract time information from metadata
        # and sort accordingly
        return sorted(sample_ids)
        
    def _get_symbol_groups(self, sample_ids: List[str]) -> List[str]:
        """
        Get symbol groups for stratification (helper method).
        
        Args:
            sample_ids: List of sample IDs
            
        Returns:
            List[str]: Symbol groups for stratification
        """
        # This is a placeholder implementation
        # In practice, you would extract symbol information from metadata
        # and return appropriate groups for stratification
        return ['default'] * len(sample_ids)
        
    def get_summary(self) -> Dict:
        """
        Get summary statistics for the dataset.
        
        Returns:
            Dict: Summary statistics
        """
        summary = {
            'total_samples': len(self.samples),
            'sample_ids': self.get_sample_ids(),
            'splits': list(self.splits.keys()),
            'data_structure': self.data_structure
        }
        
        # Add sample-specific statistics
        sample_stats = {}
        for sample_id, df in self.samples.items():
            sample_stats[sample_id] = {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            }
        summary['sample_stats'] = sample_stats
        
        return summary
        
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.samples)
        
    def __contains__(self, sample_id: str) -> bool:
        """Check if a sample exists."""
        return sample_id in self.samples 