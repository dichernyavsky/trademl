"""
Utilities for working with UniqueBarID in external datasets.
"""

import pandas as pd


def add_unique_bar_id(df: pd.DataFrame, start_id: int = 1) -> pd.DataFrame:
    """
    Add UniqueBarID as index to a DataFrame.
    
    Args:
        df: DataFrame to add UniqueBarID to
        start_id: Starting ID number (default: 1)
        
    Returns:
        DataFrame with UniqueBarID as index
    """
    df = df.copy()
    
    # Save OpenTime if index is datetime and OpenTime column doesn't exist
    if 'OpenTime' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df['OpenTime'] = df.index
    
    # Set UniqueBarID as index
    df.index = range(start_id, start_id + len(df))
    df.index.name = 'UniqueBarID'
    
    return df 