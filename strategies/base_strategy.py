from ..strategy_labeler.barriers import SimpleVolatilityBarrier
from ..indicators.strategy_indicator_manager import StrategyIndicatorManager
import pandas as pd

class BaseStrategy:
    """
    Base class for all trading strategies in the new event-based architecture.
    Now supports external barrier strategies.
    """
    
    def __init__(self, params=None, barrier_strategy=None, use_parallel_indicators=False):
        """
        Initialize strategy with parameters and barrier strategy.
        
        Args:
            params (dict, optional): Strategy parameters
            barrier_strategy (Barrier, optional): External barrier strategy
            use_parallel_indicators (bool): Whether to use parallel processing for indicators
        """
        self.params = params or {}
        self.barrier_strategy = barrier_strategy or SimpleVolatilityBarrier()
        self.indicator_manager = StrategyIndicatorManager(use_parallel=use_parallel_indicators)
        self.strategy_indicator_columns = []  # Indicators from strategy itself (like SR)
        self.init()
    
    def set_barrier_strategy(self, barrier_strategy):
        """
        Set the barrier strategy for this strategy.
        
        Args:
            barrier_strategy (Barrier): Barrier strategy to use
        """
        self.barrier_strategy = barrier_strategy
    
    def init(self):
        """
        Initialize indicators and other strategy components.
        """
        pass
    
    def add_indicators_to_data(self, data, higher_timeframe_data=None):
        """
        Add all indicators (external + strategy-specific) to data.
        This is a separate method that can be called before generate_events.
        
        Args:
            data: Original data (dict or DataFrame)
            higher_timeframe_data: Higher timeframe data for HTF indicators
            
        Returns:
            Enriched data with all indicators
        """
        # First add external indicators from indicator_manager
        if len(self.indicator_manager.indicators) > 0:
            data = self.indicator_manager.calculate_all(data, higher_timeframe_data, append=True)
        
        # Then add strategy-specific indicators
        data = self._enrich_with_strategy_indicators(data)
        
        return data
    
    def _enrich_with_strategy_indicators(self, data):
        """
        Add strategy-specific indicators.
        Override in subclasses.
        """
        return data
    
    def _add_vertical_barrier(self, events, data):
        """
        DEPRECATED: Vertical barrier logic is now handled by barrier strategies.
        This method is kept for backward compatibility but should not be used.
        """
        import warnings
        warnings.warn("_add_vertical_barrier is deprecated. Vertical barriers are now handled by barrier strategies.", 
                     DeprecationWarning, stacklevel=2)
        
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
        Generate trading events from data with barriers.
        Note: Data should already contain all necessary indicators.
        Use add_indicators_to_data() first if you need to add indicators.
        
        Args:
            data: Data with indicators already calculated
            set_barriers: Whether to set barriers
        """
        # Data should already contain all necessary indicators
        enriched_data = data
        
        # Handle different data structures
        if isinstance(enriched_data, pd.DataFrame):
            # Single DataFrame
            events = self._generate_raw_events(enriched_data)
            
            if len(events) == 0:
                return events
            
            if set_barriers:
                events = self.set_barriers(events, enriched_data)
                
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
                        
                        events_dict[interval][symbol] = events
                
                if set_barriers:
                    events_dict = self.set_barriers(events_dict, enriched_data)
                    
                return events_dict
            else:
                # Simplified format: {symbol: df}
                events_dict = {}
                for symbol, symbol_data in enriched_data.items():
                    events = self._generate_raw_events(symbol_data)
                    
                    if len(events) == 0:
                        events_dict[symbol] = events
                        continue
                    
                    events_dict[symbol] = events
                
                if set_barriers:
                    events_dict = self.set_barriers(events_dict, enriched_data)
                    
                return events_dict
        else:
            raise ValueError("Data must be DataFrame or dict of DataFrames")
    
    def generate_trades(self, data, higher_timeframe_data=None, trailing_stop=False, trailing_pct=0.0, save_trail=False, use_hl=True):
        """
        Transform events into trades with entry and exit details.
        
        Args:
            data: Data (will be enriched with indicators if needed)
            higher_timeframe_data: Higher timeframe data for HTF indicators
            trailing_stop: Whether to use trailing stop
            trailing_pct: Trailing stop percentage
            save_trail: Whether to save trailing data
            use_hl: Whether to use high/low for stops
        """
        # First, ensure data contains all necessary indicators
        enriched_data = self.add_indicators_to_data(data, higher_timeframe_data)
        
        # Generate events from enriched data
        events = self.generate_events(enriched_data, set_barriers=True)
        
        # Generate trades using barrier strategy and add indicator values
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
                        trades = self.barrier_strategy.generate_trades(
                            symbol_events, 
                            symbol_data,
                            trailing_stop=trailing_stop,
                            trailing_pct=trailing_pct,
                            save_trail=save_trail,
                            use_hl=use_hl
                        )
                        
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
                    trades = self.barrier_strategy.generate_trades(
                        symbol_events, 
                        symbol_data,
                        trailing_stop=trailing_stop,
                        trailing_pct=trailing_pct,
                        save_trail=save_trail,
                        use_hl=use_hl
                    )
                    
                    trades_dict[symbol] = trades
                    
                return trades_dict
        else:
            # Single symbol
            if len(events) == 0:
                return events  # Empty DataFrame
                
            trades = self.barrier_strategy.generate_trades(
                events, 
                enriched_data,
                trailing_stop=trailing_stop,
                trailing_pct=trailing_pct,
                save_trail=save_trail,
                use_hl=use_hl
            )
            
            return trades
    
    def generate_trades_with_analysis(self, data, trailing_stop=False, trailing_pct=0.0, 
                                    save_trail=False, use_hl=True, print_summary=True, 
                                    plot_summary=False):
        """
        Generate trades and automatically analyze performance.
        
        Args:
            data: Input data
            trailing_stop: Whether to use trailing stop
            trailing_pct: Trailing percentage
            save_trail: Whether to save trailing stop path
            use_hl: Whether to use high/low for barrier touches
            print_summary: Whether to print performance summary
            plot_summary: Whether to plot performance summary
            
        Returns:
            Tuple of (trades, analyzer)
        """
        # Generate trades
        trades = self.generate_trades(data, trailing_stop=trailing_stop, 
                                    trailing_pct=trailing_pct, save_trail=save_trail, 
                                    use_hl=use_hl)
        
        # Analyze trades
        from ..performance import TradesAnalyzer
        analyzer = TradesAnalyzer(trades)
        
        # Print summary if requested
        if print_summary:
            analyzer.print_summary()
        
        # Plot summary if requested
        if plot_summary:
            analyzer.plot_performance_summary()
        
        return trades, analyzer
    

    
    def get_all_indicator_columns(self):
        """Get all indicator column names (external + strategy)."""
        return (self.indicator_manager.get_indicator_columns() + 
                self.strategy_indicator_columns)
    
    def set_barriers(self, events, data, **kwargs):
        """
        Set profit-taking and stop-loss barriers using the configured barrier strategy.
        
        Args:
            events: Events DataFrame or dict of DataFrames
            data: Data DataFrame or dict of DataFrames
            **kwargs: Additional parameters for barrier calculation
            
        Returns:
            Events with barriers added
        """
        if isinstance(events, dict):
            # Format: {symbol: events}
            return {symbol: self.barrier_strategy.calculate_barriers(events[symbol], data[symbol], **kwargs) 
                   for symbol in events.keys() if len(events[symbol]) > 0}
        else:
            if len(events) == 0:
                return events
            return self.barrier_strategy.calculate_barriers(events, data, **kwargs)