from ..strategy_labeler.trades_generator import generate_trades
from ..strategy_labeler.barriers import ptsl_simple
from ..indicators.indicator_manager import IndicatorManager
import pandas as pd

class BaseStrategy:
    """
    Base class for all trading strategies in the new event-based architecture.
    """
    
    def __init__(self, params=None):
        """
        Initialize strategy with parameters.
        
        Args:
            params (dict, optional): Strategy parameters
        """
        self.params = params or {}
        self.indicator_manager = IndicatorManager()
        self.strategy_indicator_columns = []  # Indicators from strategy itself (like SR)
        self.init()
    
    def init(self):
        """
        Initialize indicators and other strategy components.
        """
        pass
    
    def add_external_indicators(self, data):
        """
        Add external indicators to data before strategy processing.
        This method can be called before generate_trades.
        
        Args:
            data: Original data (dict or DataFrame)
            
        Returns:
            Enriched data with external indicators
        """
        if len(self.indicator_manager.indicators) > 0:
            return self.indicator_manager.calculate_all(data, append=True)
        return data
    
    def _enrich_with_strategy_indicators(self, data):
        """
        Add strategy-specific indicators.
        Override in subclasses.
        """
        return data
    
    def _add_vertical_barrier(self, events, data):
        """Add vertical barrier (t1) handling different index types."""
        hold_periods = self.params.get('hold_periods', 50)  # Number of bars/candles to hold
        
        events['t1'] = None
        for i, event_time in enumerate(events.index):
            try:
                # Find position in data index
                event_pos = data.index.get_loc(event_time)
                # Add hold_periods positions
                target_pos = min(event_pos + hold_periods, len(data.index) - 1)
                events.iloc[i, events.columns.get_loc('t1')] = data.index[target_pos]
            except (KeyError, IndexError):
                # If event time not found or out of bounds, use last available time
                events.iloc[i, events.columns.get_loc('t1')] = data.index[-1]
        
        return events
    
    def _generate_raw_events(self, data):
        """
        Generate raw events (to be implemented by subclasses).
        This should return events without barriers or t1.
        """
        raise NotImplementedError("Subclasses must implement _generate_raw_events()")
    
    def generate_events(self, data, set_barriers=True):
        """
        Generate trading events from data with barriers and vertical barriers.
        """
        # First add external indicators
        enriched_data = self.add_external_indicators(data)
        
        # Then add strategy-specific indicators  
        enriched_data = self._enrich_with_strategy_indicators(enriched_data)
        
        # Handle different data structures
        if isinstance(enriched_data, pd.DataFrame):
            # Single DataFrame
            events = self._generate_raw_events(enriched_data)
            
            if len(events) == 0:
                return events
            
            # Add vertical barrier
            events = self._add_vertical_barrier(events, enriched_data)
            
            if set_barriers:
                events = self.set_barriers(events, enriched_data, method=self.params.get('barrier_method', 'simple'))
                
            return events
            
        elif isinstance(enriched_data, dict):
            # Check if this is the legacy format with intervals
            if any(isinstance(v, dict) for v in enriched_data.values()):
                # Legacy format: {interval: {symbol: df}}
                events_dict = {}
                for interval, symbols_data in enriched_data.items():
                    events_dict[interval] = {}
                    for symbol, symbol_data in symbols_data.items():
                        events = self._generate_raw_events(symbol_data)
                        
                        if len(events) == 0:
                            events_dict[interval][symbol] = events
                            continue
                        
                        # Add vertical barrier
                        events = self._add_vertical_barrier(events, symbol_data)
                        
                        events_dict[interval][symbol] = events
                
                if set_barriers:
                    events_dict = self.set_barriers(events_dict, enriched_data, method=self.params.get('barrier_method', 'simple'))
                    
                return events_dict
            else:
                # Simplified format: {symbol: df}
                events_dict = {}
                for symbol, symbol_data in enriched_data.items():
                    events = self._generate_raw_events(symbol_data)
                    
                    if len(events) == 0:
                        events_dict[symbol] = events
                        continue
                    
                    # Add vertical barrier
                    events = self._add_vertical_barrier(events, symbol_data)
                    
                    events_dict[symbol] = events
                
                if set_barriers:
                    events_dict = self.set_barriers(events_dict, enriched_data, method=self.params.get('barrier_method', 'simple'))
                    
                return events_dict
        else:
            raise ValueError("Data must be DataFrame or dict of DataFrames")
    
    def generate_trades(self, data, trailing_stop=False, trailing_pct=0.0, save_trail=False, use_hl=True):
        """
        Transform events into trades with entry and exit details.
        """
        # Generate events (this will enrich data with indicators)
        events = self.generate_events(data, set_barriers=True)
        
        # Get the enriched data
        enriched_data = self.add_external_indicators(data)
        enriched_data = self._enrich_with_strategy_indicators(enriched_data)
        
        # Generate trades and add indicator values
        if isinstance(events, dict):
            # Check if this is the legacy format with intervals
            if any(isinstance(v, dict) for v in events.values()):
                # Legacy format: {interval: {symbol: events}}
                trades_dict = {}
                for interval, symbols_events in events.items():
                    trades_dict[interval] = {}
                    for symbol, symbol_events in symbols_events.items():
                        if len(symbol_events) == 0:
                            trades_dict[interval][symbol] = symbol_events  # Empty DataFrame
                            continue
                            
                        symbol_data = enriched_data[interval][symbol]
                        trades = generate_trades(
                            symbol_data, 
                            symbol_events,
                            trailing_stop=trailing_stop,
                            trailing_pct=trailing_pct,
                            save_trail=save_trail,
                            use_hl=use_hl
                        )
                        
                        # Add indicator values at entry times
                        trades = self._add_indicators_to_trades(trades, symbol_data)
                        trades_dict[interval][symbol] = trades
                        
                return trades_dict
            else:
                # Simplified format: {symbol: events}
                trades_dict = {}
                for symbol, symbol_events in events.items():
                    if len(symbol_events) == 0:
                        trades_dict[symbol] = symbol_events  # Empty DataFrame
                        continue
                        
                    symbol_data = enriched_data[symbol]
                    trades = generate_trades(
                        symbol_data, 
                        symbol_events,
                        trailing_stop=trailing_stop,
                        trailing_pct=trailing_pct,
                        save_trail=save_trail,
                        use_hl=use_hl
                    )
                    
                    # Add indicator values at entry times
                    trades = self._add_indicators_to_trades(trades, symbol_data)
                    trades_dict[symbol] = trades
                    
                return trades_dict
        else:
            # Single symbol
            if len(events) == 0:
                return events  # Empty DataFrame
                
            trades = generate_trades(
                enriched_data, 
                events,
                trailing_stop=trailing_stop,
                trailing_pct=trailing_pct,
                save_trail=save_trail,
                use_hl=use_hl
            )
            
            # Add indicator values at entry times
            trades = self._add_indicators_to_trades(trades, enriched_data)
            return trades
    
    def _add_indicators_to_trades(self, trades, enriched_data):
        """Add indicator values to trades at entry times."""
        if len(trades) == 0:
            return trades
            
        # Get all indicator columns
        all_indicator_cols = (self.indicator_manager.get_indicator_columns() + 
                             self.strategy_indicator_columns)
        
        # Add indicator values at entry times
        for col in all_indicator_cols:
            if col in enriched_data.columns:
                trades[col] = enriched_data.loc[trades.index, col]
        
        return trades
    
    def get_all_indicator_columns(self):
        """Get all indicator column names (external + strategy)."""
        return (self.indicator_manager.get_indicator_columns() + 
                self.strategy_indicator_columns)
    
    def set_barriers(self, events, data, method='simple', **kwargs):
        """Set profit-taking and stop-loss barriers for events."""
        if isinstance(events, dict):
            # Check if this is the legacy format with intervals
            if any(isinstance(v, dict) for v in events.values()):
                # Legacy format: {interval: {symbol: events}}
                result = {}
                for interval, symbols_events in events.items():
                    result[interval] = {}
                    for symbol, symbol_events in symbols_events.items():
                        if len(symbol_events) > 0:
                            result[interval][symbol] = self._set_barriers_single(
                                symbol_events, data[interval][symbol], method, **kwargs
                            )
                        else:
                            result[interval][symbol] = symbol_events
                return result
            else:
                # Simplified format: {symbol: events}
                return {symbol: self._set_barriers_single(events[symbol], data[symbol], method, **kwargs) 
                       for symbol in events.keys() if len(events[symbol]) > 0}
        else:
            if len(events) == 0:
                return events
            return self._set_barriers_single(events, data, method, **kwargs)
    
    def _set_barriers_single(self, events, data, method, **kwargs):
        """Set barriers for single symbol."""
        # Merge params with kwargs, giving priority to kwargs
        barrier_params = {**self.params, **kwargs}
        
        if method == 'simple':
            return ptsl_simple(
                events, 
                data['Close'], 
                window=barrier_params.get('window', 20),
                multiplier=barrier_params.get('multiplier', [2, 2]),
                min_ret=barrier_params.get('min_ret', 0.001)
            )
        else:
            raise ValueError(f"Unknown barrier method: {method}")