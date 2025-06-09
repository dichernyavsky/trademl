"""
Base classes for event generators used in labeling.
"""

import pandas as pd
import numpy as np
from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional, List, Union, Dict, Any, Tuple

class EventType(Enum):
    """
    Enum for different types of events that can be generated.
    """
    DIRECTION_AGNOSTIC = "direction_agnostic"   # Example: volatility spikes
    DIRECTION_SPECIFIC = "direction_specific"   # Example: price breakouts
   
class EventGenerator(ABC):
    """
    Abstract base class for all event generators.
    
    An event generator identifies points of interest in financial data,
    which can be used for labeling and strategy development.
    """
    
    def __init__(self, event_type: EventType = EventType.DIRECTION_AGNOSTIC):
        """
        Initialize the event generator.
        
        Parameters
        ----------
        event_type : EventType
            The type of events this generator produces
        """
        self.event_type = event_type
        self.events = None
        self.indicators = {}  # Dictionary of indicator objects (name -> indicator)
    
    @abstractmethod
    def generate(self, **kwargs) -> Union[pd.DatetimeIndex, pd.Series, pd.DataFrame]:
        """
        Generate events based on the implementation criteria.
        
        Returns
        -------
        Union[pd.DatetimeIndex, pd.Series, pd.DataFrame]
            - DatetimeIndex containing event timestamps (for direction_agnostic events)
            - Series with direction values (for direction_specific events)
            - DataFrame with timestamps as index and direction column (for more complex events)
        """
        pass
    
    def get_events(self) -> Union[pd.DatetimeIndex, pd.Series, pd.DataFrame]:
        """
        Return previously generated events.
        
        Returns
        -------
        Union[pd.DatetimeIndex, pd.Series, pd.DataFrame]
            Generated events with direction information if applicable
        
        Raises
        ------
        ValueError
            If no events have been generated yet
        """
        if self.events is None:
            raise ValueError("No events generated yet. Call generate() first.")
        return self.events
    
    def add_indicator(self, indicator):
        """
        Add an indicator to this generator.
        
        Parameters
        ----------
        indicator : BaseIndicator
            Technical indicator to add
        """
        self.indicators[indicator.name] = indicator
    
    def get_indicator(self, name):
        """
        Get a specific indicator by name.
        
        Parameters
        ----------
        name : str
            Name of the indicator to retrieve
            
        Returns
        -------
        BaseIndicator
            The requested indicator
            
        Raises
        ------
        KeyError
            If no indicator with the given name exists
        """
        if name not in self.indicators:
            raise KeyError(f"No indicator named '{name}' found")
        return self.indicators[name]
    
    def get_indicators(self):
        """
        Get all indicators used by this generator.
        
        Returns
        -------
        dict
            Dictionary mapping indicator names to indicator objects
        """
        return self.indicators
    
    def calculate_indicators(self, data, append_indicators=True):
        """
        Calculate all indicators for this generator.
        
        Parameters
        ----------
        data : pd.DataFrame
            OHLCV data to calculate indicators on
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing all indicator results
        """
        # Create a copy of the input data to store results
        results = pd.DataFrame(index=data.index)
        
        for name, indicator in self.indicators.items():
            # Use the indicator's calculate method
            indicator_result = indicator.calculate(data, append_indicators)
            
            # If indicator returns a DataFrame with multiple columns, add all columns
            if isinstance(indicator_result, pd.DataFrame):
                for column in indicator_result.columns:
                    results[column] = indicator_result[column]
            else:
                # Single column result
                results[name] = indicator_result
            
        return results
    
    def is_directional(self) -> bool:
        """Return whether this generator provides directional information."""
        return self.event_type == EventType.DIRECTION_SPECIFIC
    
    def __str__(self):
        return self.__class__.__name__