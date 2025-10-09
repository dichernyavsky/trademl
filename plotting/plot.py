from __future__ import annotations

import os
import warnings
from colorsys import hls_to_rgb, rgb_to_hls
from itertools import cycle
from functools import partial
from typing import List, Union, Optional, Dict, Any

import numpy as np
import pandas as pd
from bokeh.colors import RGB
from bokeh.colors.named import lime as BULL_COLOR, tomato as BEAR_COLOR
from bokeh.plotting import figure as _figure
from bokeh.models import (
    CrosshairTool, CustomJS, ColumnDataSource, Span, Label,
    NumeralTickFormatter, HoverTool, Range1d, DatetimeTickFormatter,
    WheelZoomTool, CustomJSTickFormatter
)
from bokeh.io import output_notebook, output_file, show
from bokeh.layouts import gridplot
from bokeh.palettes import Category10
from bokeh.transform import factor_cmap

# Check if running in Jupyter notebook
IS_JUPYTER_NOTEBOOK = ('JPY_PARENT_PID' in os.environ or
                      'inline' in os.environ.get('MPLBACKEND', ''))
if IS_JUPYTER_NOTEBOOK:
    output_notebook()

def set_bokeh_output(notebook=False):
    """
    Set Bokeh to output either to a file or Jupyter notebook.
    """
    global IS_JUPYTER_NOTEBOOK
    IS_JUPYTER_NOTEBOOK = notebook
    if notebook:
        output_notebook()

def _windows_safe_filename(filename):
    """Create a Windows-safe filename."""
    import re
    import sys
    if sys.platform.startswith('win'):
        return re.sub(r'[^a-zA-Z0-9,_-]', '_', filename.replace('=', '-'))
    return filename

def _bokeh_reset(filename=None):
    """Reset Bokeh state and set output."""
    from bokeh.io.state import curstate
    curstate().reset()
    if filename:
        if not filename.endswith('.html'):
            filename += '.html'
        output_file(filename, title=filename)
    elif IS_JUPYTER_NOTEBOOK:
        output_notebook()

def _watermark(fig):
    """Add a watermark to the figure."""
    fig.add_layout(
        Label(
            x=10, y=15, x_units='screen', y_units='screen', text_color='silver',
            text='Created with Market Data Plotter',
            text_alpha=.09))

def colorgen():
    """Generate colors from Category10 palette."""
    yield from cycle(Category10[10])

def lightness(color, lightness=.94):
    """Adjust the lightness of a color."""
    rgb = np.array([color.r, color.g, color.b]) / 255
    h, _, s = rgb_to_hls(*rgb)
    rgb = (np.array(hls_to_rgb(h, lightness, s)) * 255).astype(int)
    return RGB(*rgb)

class Indicator:
    """Simple indicator class to hold data and plotting options."""
    def __init__(self, data, name=None, overlay=False, color=None, scatter=False, plot=True, subplot_group=None):
        self.data = np.asarray(data)
        self.name = name
        self.overlay = overlay
        self.color = color
        self.scatter = scatter
        self.plot = plot
        self.subplot_group = subplot_group  # Group indicators on same subplot

def plot_market_data(
    df: pd.DataFrame,
    indicators: List[Indicator] = None,
    events: pd.DataFrame = None,
    signals: pd.DataFrame = None,
    signals_config: Dict[str, Dict[str, str]] = None,
    filename: str = '',
    plot_width: int = None,
    plot_volume: bool = True,
    show_legend: bool = True,
    open_browser: bool = True
):
    """
    Plot OHLCV data and indicators.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC(V) data. Must have columns: Open, High, Low, Close, and optionally Volume.
    indicators : List[Indicator]
        List of Indicator objects to plot.
    events : pd.DataFrame
        DataFrame of events with index as timestamps and columns that may include 'direction' and 'entry_price'.
    signals : pd.DataFrame
        DataFrame of signals with index as timestamps and columns containing signal values (0, 1, -1).
        Each column will be plotted as a separate signal type.
    signals_config : Dict[str, Dict[str, str]]
        Configuration for signals plotting. Format:
        {
            'signal_column_name': {
                'values_col': 'column_with_0_1_-1_values',
                'price_col': 'column_with_price_values'  # Optional, defaults to 'Close'
            }
        }
        If None, uses default behavior (all columns as signals, Close price for Y position).
    filename : str
        If specified, the plot is saved to this file.
    plot_width : int
        Width of the plot in pixels.
    plot_volume : bool
        Whether to plot volume.
    show_legend : bool
        Whether to show the legend.
    open_browser : bool
        Whether to open the browser when saving to file.
    """
    if indicators is None:
        indicators = []
        
    # Reset Bokeh state
    _bokeh_reset(filename)
    
    # Constants
    COLORS = [BEAR_COLOR, BULL_COLOR]
    BAR_WIDTH = .8
    NBSP = '\N{NBSP}' * 4
    
    # Check if we have a datetime index or OpenTime column
    is_datetime_index = isinstance(df.index, pd.DatetimeIndex)
    has_open_time = 'OpenTime' in df.columns
    
    # Ensure we have the required columns
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {', '.join(required_cols)}")
    
    # Check if we should plot volume
    plot_volume = plot_volume and 'Volume' in df.columns and not df.Volume.isnull().all()
    
    # Prepare data
    df = df.copy()
    if is_datetime_index:
        df['datetime'] = df.index  # Save original datetime index
    elif has_open_time:
        df['datetime'] = df['OpenTime']  # Use OpenTime column for datetime
    else:
        # If no datetime info, use index as x-axis
        df['datetime'] = df.index
    
    df = df.reset_index(drop=True)
    index = df.index
    
    # Create figure
    new_bokeh_figure = partial(
        _figure,
        x_axis_type='linear',
        width=plot_width,
        height=400,
        tools="xpan,xwheel_zoom,xwheel_pan,box_zoom,undo,redo,reset,save",
        active_drag='xpan',
        active_scroll='xwheel_zoom'
    )
    
    # Set x-range padding
    pad = (index[-1] - index[0]) / 20 if len(index) > 1 else 0
    kwargs = dict(
        x_range=Range1d(
            index[0], index[-1],
            min_interval=10,
            bounds=(index[0] - pad, index[-1] + pad)
        )
    ) if len(index) > 1 else {}
    
    # Create main OHLC figure
    fig_ohlc = new_bokeh_figure(**kwargs)
    
    # Lists to hold figures above and below the main OHLC chart
    figs_above_ohlc, figs_below_ohlc = [], []
    
    # Create data source
    source = ColumnDataSource(df)
    source.add((df.Close >= df.Open).values.astype(np.uint8).astype(str), 'inc')
    
    # Create color maps
    inc_cmap = factor_cmap('inc', COLORS, ['0', '1'])
    
    # Set datetime formatter if we have datetime data
    if is_datetime_index or has_open_time:
        fig_ohlc.xaxis.formatter = CustomJSTickFormatter(
            args=dict(
                axis=fig_ohlc.xaxis[0],
                formatter=DatetimeTickFormatter(
                    days='%a, %d %b',
                    months='%m/%Y'
                ),
                source=source
            ),
            code='''
            this.labels = this.labels || formatter.doFormat(ticks
                .map(i => source.data.datetime[i])
                .filter(t => t !== undefined));
            return this.labels[index] || "";
            '''
        )
    
    # Prepare for tooltips
    ohlc_extreme_values = df[['High', 'Low']].copy()
    ohlc_tooltips = [
        ('x, y', NBSP.join(('$index', '$y{0,0.0[0000]}'))),
        ('OHLC', NBSP.join(('@Open{0,0.0[0000]}', '@High{0,0.0[0000]}',
                           '@Low{0,0.0[0000]}', '@Close{0,0.0[0000]}'))),
    ]
    
    if plot_volume:
        ohlc_tooltips.append(('Volume', '@Volume{0,0}'))
    
    # Helper function to create indicator figures
    def new_indicator_figure(**kwargs):
        kwargs.setdefault('height', 150)
        fig = new_bokeh_figure(
            x_range=fig_ohlc.x_range,
            active_scroll='xwheel_zoom',
            active_drag='xpan',
            **kwargs
        )
        fig.xaxis.visible = False
        fig.yaxis.minor_tick_line_color = None
        fig.yaxis.ticker.desired_num_ticks = 3
        return fig
    
    # Helper function to set tooltips
    def set_tooltips(fig, tooltips=(), vline=True, renderers=()):
        tooltips = list(tooltips)
        renderers = list(renderers)
        
        if is_datetime_index:
            formatters = {'@datetime': 'datetime'}
            tooltips = [("Date", "@datetime{%c}")] + tooltips
        else:
            formatters = {}
            tooltips = [("#", "@index")] + tooltips
            
        fig.add_tools(HoverTool(
            point_policy='follow_mouse',
            renderers=renderers, formatters=formatters,
            tooltips=tooltips, mode='vline' if vline else 'mouse'
        ))
    
    # Plot volume section
    def _plot_volume_section():
        fig = new_indicator_figure(height=100, y_axis_label="Volume")
        fig.yaxis.ticker.desired_num_ticks = 3
        fig.xaxis.formatter = fig_ohlc.xaxis[0].formatter
        fig.xaxis.visible = True
        fig_ohlc.xaxis.visible = False  # Show only Volume's xaxis
        
        r = fig.vbar('index', BAR_WIDTH, 'Volume', source=source, color=inc_cmap)
        set_tooltips(fig, [('Volume', '@Volume{0.00 a}')], renderers=[r])
        fig.yaxis.formatter = NumeralTickFormatter(format="0 a")
        return fig
    
    # Plot OHLC bars
    def _plot_ohlc():
        fig_ohlc.segment('index', 'High', 'index', 'Low', source=source, color="black",
                        legend_label='OHLC')
        r = fig_ohlc.vbar('index', BAR_WIDTH, 'Open', 'Close', source=source,
                         line_color="black", fill_color=inc_cmap, legend_label='OHLC')
        return r
    
    # Plot barriers (Take Profit and Stop Loss)
    def _plot_barriers():
        if events is None or events.empty:
            return None
            
        # Get the original datetime index if it exists
        original_datetime_index = None
        if 'datetime' in df.columns:
            original_datetime_index = df['datetime']
        
        # Collect barrier data
        barrier_times = []
        barrier_prices = []
        barrier_types = []
        barrier_tooltips = []
        
        for event_time in events.index:
            event_row = events.loc[event_time]
            
            # Find corresponding index in main dataframe
            idx = None
            
            # Method 1: Try to find in original datetime index if available
            if original_datetime_index is not None:
                try:
                    datetime_positions = original_datetime_index[original_datetime_index == event_time]
                    if len(datetime_positions) > 0:
                        idx = datetime_positions.index[0]
                except:
                    pass
            
            # Method 2: Try to find in current df index (if it's still datetime)
            if idx is None and isinstance(df.index, pd.DatetimeIndex):
                if event_time in df.index:
                    idx = df.index.get_loc(event_time)
            
            # Method 3: Try nearest match if we have datetime data
            if idx is None and (original_datetime_index is not None or isinstance(df.index, pd.DatetimeIndex)):
                try:
                    if original_datetime_index is not None:
                        nearest_idx = original_datetime_index.index[original_datetime_index.searchsorted(event_time)]
                        idx = nearest_idx
                    else:
                        nearest_idx = df.index.searchsorted(event_time)
                        if nearest_idx < len(df.index):
                            idx = nearest_idx
                except:
                    pass
            
            # Method 4: Fallback
            if idx is None:
                event_position = events.index.get_loc(event_time)
                if event_position < len(df):
                    idx = event_position
            
            if idx is None:
                continue
            
            # Add Take Profit barrier
            if 'pt' in events.columns and not pd.isna(event_row.get('pt')):
                barrier_times.append(idx)
                barrier_prices.append(event_row['pt'])
                barrier_types.append('TP')
                barrier_tooltips.append(f"Take Profit: {event_row['pt']:.2f}")
            
            # Add Stop Loss barrier
            if 'sl' in events.columns and not pd.isna(event_row.get('sl')):
                barrier_times.append(idx)
                barrier_prices.append(event_row['sl'])
                barrier_types.append('SL')
                barrier_tooltips.append(f"Stop Loss: {event_row['sl']:.2f}")
        
        if not barrier_times:
            return None
            
        # Create barrier data source
        barrier_source = ColumnDataSource({
            'x': barrier_times,
            'y': barrier_prices,
            'type': barrier_types,
            'tooltip': barrier_tooltips
        })
        
        # Plot Take Profit barriers
        tp_indices = [i for i, t in enumerate(barrier_types) if t == 'TP']
        if tp_indices:
            tp_source = ColumnDataSource({
                'x': [barrier_times[i] for i in tp_indices],
                'y': [barrier_prices[i] for i in tp_indices],
                'tooltip': [barrier_tooltips[i] for i in tp_indices]
            })
            r_tp = fig_ohlc.diamond(
                x='x', y='y', size=8, color='green', alpha=0.8,
                line_color='darkgreen', line_width=2, source=tp_source,
                legend_label='Take Profit'
            )
            fig_ohlc.add_tools(HoverTool(
                renderers=[r_tp],
                tooltips=[('Take Profit', '@tooltip')],
                point_policy='follow_mouse',
                mode='mouse'
            ))
        
        # Plot Stop Loss barriers
        sl_indices = [i for i, t in enumerate(barrier_types) if t == 'SL']
        if sl_indices:
            sl_source = ColumnDataSource({
                'x': [barrier_times[i] for i in sl_indices],
                'y': [barrier_prices[i] for i in sl_indices],
                'tooltip': [barrier_tooltips[i] for i in sl_indices]
            })
            r_sl = fig_ohlc.diamond(
                x='x', y='y', size=8, color='red', alpha=0.8,
                line_color='darkred', line_width=2, source=sl_source,
                legend_label='Stop Loss'
            )
            fig_ohlc.add_tools(HoverTool(
                renderers=[r_sl],
                tooltips=[('Stop Loss', '@tooltip')],
                point_policy='follow_mouse',
                mode='mouse'
            ))
    
    # Plot events
    def _plot_events():
        if events is None or events.empty:
            return None
            
        # Merge events with main dataframe to get the right indices
        event_indices = []
        event_y_values = []
        event_directions = []
        event_tooltips = []
        
        # Get the original datetime index if it exists
        original_datetime_index = None
        if 'datetime' in df.columns:
            original_datetime_index = df['datetime']
        
        for event_time in events.index:
            # Try to find the corresponding index in the main dataframe
            idx = None
            
            # Method 1: Try to find in original datetime index if available
            if original_datetime_index is not None:
                try:
                    # Find the position in the original datetime series
                    datetime_positions = original_datetime_index[original_datetime_index == event_time]
                    if len(datetime_positions) > 0:
                        # Get the position in the current df
                        idx = datetime_positions.index[0]
                except:
                    pass
            
            # Method 2: Try to find in current df index (if it's still datetime)
            if idx is None and isinstance(df.index, pd.DatetimeIndex):
                if event_time in df.index:
                    idx = df.index.get_loc(event_time)
            
            # Method 3: Try nearest match if we have datetime data
            if idx is None and (original_datetime_index is not None or isinstance(df.index, pd.DatetimeIndex)):
                try:
                    if original_datetime_index is not None:
                        # Find nearest in original datetime
                        nearest_idx = original_datetime_index.index[original_datetime_index.searchsorted(event_time)]
                        idx = nearest_idx
                    else:
                        # Find nearest in current df index
                        nearest_idx = df.index.searchsorted(event_time)
                        if nearest_idx < len(df.index):
                            idx = nearest_idx
                except:
                    pass
            
            # Method 4: If all else fails, try to find by position in events
            if idx is None:
                # This is a fallback - try to map event position to df position
                event_position = events.index.get_loc(event_time)
                if event_position < len(df):
                    idx = event_position
            
            if idx is None:
                continue
                
            event_indices.append(idx)
            
            # Determine Y position (entry_price or Close)
            event_row = events.loc[event_time]
            if 'entry_price' in events.columns and not pd.isna(event_row.get('entry_price')):
                y_value = event_row['entry_price']
            else:
                y_value = df.iloc[idx]['Close']
                
            event_y_values.append(y_value)
            
            # Get direction for color
            if 'direction' in events.columns:
                direction = event_row['direction']
                if pd.isna(direction):
                    direction = 0
            else:
                direction = 0
                
            event_directions.append(int(direction))
            
            # Create tooltip text
            tooltip = []
            for col in events.columns:
                if col != 'direction':
                    value = event_row[col]
                    if not pd.isna(value):
                        tooltip.append(f"{col}: {value}")
            
            event_tooltips.append(", ".join(tooltip))
        
        if not event_indices:
            return None
            
        # Create event data source
        event_source = ColumnDataSource({
            'x': event_indices,
            'y': event_y_values,
            'direction': event_directions,
            'tooltip': event_tooltips
        })
        
        # Plot buy signals (direction = 1)
        buy_indices = [i for i, d in enumerate(event_directions) if d == 1]
        if buy_indices:
            buy_source = ColumnDataSource({
                'x': [event_indices[i] for i in buy_indices],
                'y': [event_y_values[i] for i in buy_indices],
                'tooltip': [event_tooltips[i] for i in buy_indices]
            })
            r_buy = fig_ohlc.triangle(
                x='x', y='y', size=12, color='lime', alpha=0.8,
                line_color='black', line_width=1, source=buy_source,
                legend_label='Buy Signal'
            )
            fig_ohlc.add_tools(HoverTool(
                renderers=[r_buy],
                tooltips=[('Buy Signal', '@tooltip')],
                point_policy='follow_mouse',
                mode='mouse'
            ))
        
        # Plot sell signals (direction = -1)
        sell_indices = [i for i, d in enumerate(event_directions) if d == -1]
        if sell_indices:
            sell_source = ColumnDataSource({
                'x': [event_indices[i] for i in sell_indices],
                'y': [event_y_values[i] for i in sell_indices],
                'tooltip': [event_tooltips[i] for i in sell_indices]
            })
            r_sell = fig_ohlc.inverted_triangle(
                x='x', y='y', size=12, color='tomato', alpha=0.8,
                line_color='black', line_width=1, source=sell_source,
                legend_label='Sell Signal'
            )
            fig_ohlc.add_tools(HoverTool(
                renderers=[r_sell],
                tooltips=[('Sell Signal', '@tooltip')],
                point_policy='follow_mouse',
                mode='mouse'
            ))
        
        # Plot neutral signals (direction = 0)
        neutral_indices = [i for i, d in enumerate(event_directions) if d == 0]
        if neutral_indices:
            neutral_source = ColumnDataSource({
                'x': [event_indices[i] for i in neutral_indices],
                'y': [event_y_values[i] for i in neutral_indices],
                'tooltip': [event_tooltips[i] for i in neutral_indices]
            })
            r_neutral = fig_ohlc.circle(
                x='x', y='y', size=8, color='blue', alpha=0.8,
                line_color='black', line_width=1, source=neutral_source,
                legend_label='Event'
            )
            fig_ohlc.add_tools(HoverTool(
                renderers=[r_neutral],
                tooltips=[('Event', '@tooltip')],
                point_policy='follow_mouse',
                mode='mouse'
            ))
    
    # Plot signals
    def _plot_signals():
        if signals is None or signals.empty:
            return None
            
        # Get the original datetime index if it exists
        original_datetime_index = None
        if 'datetime' in df.columns:
            original_datetime_index = df['datetime']
        
        # Determine which columns to process
        if signals_config is not None:
            # Use configuration to determine which columns to plot
            columns_to_process = signals_config.keys()
        else:
            # Default behavior: process all columns
            columns_to_process = signals.columns
        
        # Process each signal column
        for signal_col in columns_to_process:
            if signals_config is not None:
                # Use configuration
                config = signals_config[signal_col]
                values_col = config['values_col']
                price_col = config.get('price_col', 'Close')
                
                if values_col not in signals.columns:
                    print(f"Warning: values column '{values_col}' not found in signals DataFrame")
                    continue
                    
                signal_series = signals[values_col]
                
                # Get price values
                if price_col == 'Close':
                    # Use Close price from main dataframe
                    price_series = None
                elif price_col in signals.columns:
                    # Use price from signals DataFrame
                    price_series = signals[price_col]
                else:
                    print(f"Warning: price column '{price_col}' not found, using Close price")
                    price_series = None
            else:
                # Default behavior
                signal_series = signals[signal_col]
                price_series = None
                price_col = 'Close'
            
            # Find indices where signal is not 0
            non_zero_signals = signal_series[signal_series != 0]
            
            if len(non_zero_signals) == 0:
                continue
                
            signal_indices = []
            signal_y_values = []
            signal_values = []
            signal_tooltips = []
            
            for signal_time in non_zero_signals.index:
                # Try to find the corresponding index in the main dataframe
                idx = None
                
                # Method 1: Try to find in original datetime index if available
                if original_datetime_index is not None:
                    try:
                        datetime_positions = original_datetime_index[original_datetime_index == signal_time]
                        if len(datetime_positions) > 0:
                            idx = datetime_positions.index[0]
                    except:
                        pass
                
                # Method 2: Try to find in current df index (if it's still datetime)
                if idx is None and isinstance(df.index, pd.DatetimeIndex):
                    if signal_time in df.index:
                        idx = df.index.get_loc(signal_time)
                
                # Method 3: Try nearest match if we have datetime data
                if idx is None and (original_datetime_index is not None or isinstance(df.index, pd.DatetimeIndex)):
                    try:
                        if original_datetime_index is not None:
                            nearest_idx = original_datetime_index.index[original_datetime_index.searchsorted(signal_time)]
                            idx = nearest_idx
                        else:
                            nearest_idx = df.index.searchsorted(signal_time)
                            if nearest_idx < len(df.index):
                                idx = nearest_idx
                    except:
                        pass
                
                # Method 4: If all else fails, try to find by position
                if idx is None:
                    signal_position = signal_series.index.get_loc(signal_time)
                    if signal_position < len(df):
                        idx = signal_position
                
                if idx is None:
                    continue
                    
                signal_indices.append(idx)
                
                # Determine Y position
                if price_series is not None and signal_time in price_series.index:
                    y_value = price_series[signal_time]
                else:
                    y_value = df.iloc[idx]['Close']
                    
                signal_y_values.append(y_value)
                # Handle NaN values before converting to int
                signal_val = non_zero_signals[signal_time]
                if pd.isna(signal_val):
                    continue  # Skip NaN values
                signal_values.append(int(signal_val))
                
                # Create tooltip
                if signals_config is not None:
                    tooltip = f"{signal_col}: {signal_val} (price: {y_value:.4f})"
                else:
                    tooltip = f"{signal_col}: {signal_val}"
                signal_tooltips.append(tooltip)
            
            if not signal_indices:
                continue
                
            # Create signal data source
            signal_source = ColumnDataSource({
                'x': signal_indices,
                'y': signal_y_values,
                'signal': signal_values,
                'tooltip': signal_tooltips
            })
            
            # Plot positive signals (signal = 1)
            pos_indices = [i for i, s in enumerate(signal_values) if s == 1]
            if pos_indices:
                pos_source = ColumnDataSource({
                    'x': [signal_indices[i] for i in pos_indices],
                    'y': [signal_y_values[i] for i in pos_indices],
                    'tooltip': [signal_tooltips[i] for i in pos_indices]
                })
                r_pos = fig_ohlc.diamond(
                    x='x', y='y', size=10, color='green', alpha=0.8,
                    line_color='black', line_width=1, source=pos_source,
                    legend_label=f'{signal_col} (Positive)'
                )
                fig_ohlc.add_tools(HoverTool(
                    renderers=[r_pos],
                    tooltips=[(f'{signal_col} (Positive)', '@tooltip')],
                    point_policy='follow_mouse',
                    mode='mouse'
                ))
            
            # Plot negative signals (signal = -1)
            neg_indices = [i for i, s in enumerate(signal_values) if s == -1]
            if neg_indices:
                neg_source = ColumnDataSource({
                    'x': [signal_indices[i] for i in neg_indices],
                    'y': [signal_y_values[i] for i in neg_indices],
                    'tooltip': [signal_tooltips[i] for i in neg_indices]
                })
                r_neg = fig_ohlc.diamond(
                    x='x', y='y', size=10, color='red', alpha=0.8,
                    line_color='black', line_width=1, source=neg_source,
                    legend_label=f'{signal_col} (Negative)'
                )
                fig_ohlc.add_tools(HoverTool(
                    renderers=[r_neg],
                    tooltips=[(f'{signal_col} (Negative)', '@tooltip')],
                    point_policy='follow_mouse',
                    mode='mouse'
                ))
    
    # Plot indicators
    def _plot_indicators():
        class LegendStr(str):
            # Ensures unique legend items even with same string content
            def __eq__(self, other):
                return self is other
        
        ohlc_colors = colorgen()
        indicator_figs = []
        
        # Group indicators by subplot_group
        grouped_indicators = {}
        for i, ind in enumerate(indicators):
            if not ind.plot:
                continue
                
            if ind.overlay:
                # Overlay indicators go on main chart
                group_key = 'overlay'
            elif ind.subplot_group is not None:
                # Grouped indicators share a subplot
                group_key = f'group_{ind.subplot_group}'
            else:
                # Each indicator gets its own subplot
                group_key = f'single_{i}'
                
            if group_key not in grouped_indicators:
                grouped_indicators[group_key] = []
            grouped_indicators[group_key].append((i, ind))
        
        # Process each group
        for group_key, group_indicators in grouped_indicators.items():
            if group_key == 'overlay':
                fig = fig_ohlc
            else:
                fig = new_indicator_figure()
                indicator_figs.append(fig)
            
            # Process all indicators in this group
            for i, ind in group_indicators:
                value = np.atleast_2d(ind.data)
                
                tooltips = []
                colors = ind.color
                colors = colors and cycle([colors]) or (
                    cycle([next(ohlc_colors)]) if ind.overlay else colorgen())
                    
                if isinstance(ind.name, str):
                    tooltip_label = ind.name
                    legend_labels = [LegendStr(ind.name)] * len(value)
                else:
                    tooltip_label = ", ".join(ind.name)
                    legend_labels = [LegendStr(item) for item in ind.name]
                    
                for j, arr in enumerate(value):
                    color = next(colors)
                    source_name = f'{legend_labels[j]}_{i}_{j}'
                    
                    if arr.dtype == bool:
                        arr = arr.astype(int)
                        
                    source.add(arr, source_name)
                    tooltips.append(f'@{{{source_name}}}{{0,0.0[0000]}}')
                    
                    if ind.overlay:
                        ohlc_extreme_values[source_name] = arr
                        
                    if ind.scatter:
                        fig.circle(
                            'index', source_name, source=source,
                            legend_label=legend_labels[j], color=color,
                            line_color='black', fill_alpha=.8,
                            radius=BAR_WIDTH / 2 * .9
                        )
                    else:
                        fig.line(
                            'index', source_name, source=source,
                            legend_label=legend_labels[j], line_color=color,
                            line_width=1.3
                        )
                        
                    # Add mean line if appropriate
                    mean = np.nanmean(arr) if not np.all(np.isnan(arr)) else np.nan
                    if not np.isnan(mean) and (abs(mean) < .1 or
                                              round(abs(mean), 1) == .5 or
                                              round(abs(mean), -1) in (50, 100, 200)):
                        fig.add_layout(Span(
                            location=float(mean), dimension='width',
                            line_color='#666666', line_dash='dashed',
                            level='underlay', line_width=.5
                        ))
                        
                    if ind.overlay:
                        ohlc_tooltips.append((tooltip_label, NBSP.join(tooltips)))
                    else:
                        r = fig.line('index', source_name, source=source, line_color=color, line_width=1.3)
                        set_tooltips(fig, [(tooltip_label, NBSP.join(tooltips))], vline=True, renderers=[r])
                        
                # If the sole indicator line on this figure,
                # have the legend only contain text without the glyph
                if len(value) == 1 and len(group_indicators) == 1:
                    fig.legend.glyph_width = 0
                    
        return indicator_figs
    
    # Plot the components
    if plot_volume:
        fig_volume = _plot_volume_section()
        figs_below_ohlc.append(fig_volume)
        
    ohlc_bars = _plot_ohlc()
    
    # Plot events on the OHLC chart
    _plot_events()

    # Plot signals on the OHLC chart
    _plot_signals()

    # Plot barriers on the OHLC chart
    _plot_barriers()
    
    indicator_figs = _plot_indicators()
    figs_below_ohlc.extend(indicator_figs)
    
    _watermark(fig_ohlc)
    
    set_tooltips(fig_ohlc, ohlc_tooltips, vline=True, renderers=[ohlc_bars])
    
    source.add(ohlc_extreme_values.min(1), 'ohlc_low')
    source.add(ohlc_extreme_values.max(1), 'ohlc_high')
    
    # Auto-scale Y axis
    custom_js_args = dict(ohlc_range=fig_ohlc.y_range, source=source)
    if plot_volume:
        custom_js_args.update(volume_range=fig_volume.y_range)
        
    # JavaScript callback for auto-scaling
    autoscale_js = """
    const start = cb_obj.start;
    const end = cb_obj.end;
    
    // Find data range
    let min = Infinity;
    let max = -Infinity;
    
    for (let i = Math.floor(start); i <= Math.ceil(end); i++) {
        if (i >= 0 && i < source.data.index.length) {
            if (source.data.ohlc_low[i] < min) min = source.data.ohlc_low[i];
            if (source.data.ohlc_high[i] > max) max = source.data.ohlc_high[i];
        }
    }
    
    // Add some padding
    const padding = (max - min) * 0.1;
    ohlc_range.start = min - padding;
    ohlc_range.end = max + padding;
    
    // Also adjust volume if present
    if (volume_range !== undefined) {
        let vol_max = -Infinity;
        for (let i = Math.floor(start); i <= Math.ceil(end); i++) {
            if (i >= 0 && i < source.data.index.length) {
                if (source.data.Volume[i] > vol_max) vol_max = source.data.Volume[i];
            }
        }
        volume_range.start = 0;
        volume_range.end = vol_max * 1.1;
    }
    """
    
    fig_ohlc.x_range.js_on_change('end', CustomJS(args=custom_js_args, code=autoscale_js))
    
    # Combine all figures
    figs = figs_above_ohlc + [fig_ohlc] + figs_below_ohlc
    
    # Create linked crosshair
    linked_crosshair = CrosshairTool(
        dimensions='both', line_color='lightgrey',
        overlay=(
            Span(dimension="width", line_dash="dotted", line_width=1),
            Span(dimension="height", line_dash="dotted", line_width=1)
        ),
    )
    
    # Apply common styling to all figures
    for f in figs:
        if f.legend:
            f.legend.visible = show_legend
            f.legend.location = 'top_left'
            f.legend.border_line_width = 1
            f.legend.border_line_color = '#333333'
            f.legend.padding = 5
            f.legend.spacing = 0
            f.legend.margin = 0
            f.legend.label_text_font_size = '8pt'
            f.legend.click_policy = "hide"
            f.legend.background_fill_alpha = .9
            
        f.min_border_left = 0
        f.min_border_top = 3
        f.min_border_bottom = 6
        f.min_border_right = 10
        f.outline_line_color = '#666666'
        f.add_tools(linked_crosshair)
        
        wheelzoom_tool = next(wz for wz in f.tools if isinstance(wz, WheelZoomTool))
        wheelzoom_tool.maintain_focus = False
    
    # Create the grid layout
    kwargs = {}
    if plot_width is None:
        kwargs['sizing_mode'] = 'stretch_width'
        
    fig = gridplot(
        figs,
        ncols=1,
        toolbar_location='right',
        toolbar_options=dict(logo=None),
        merge_tools=True,
        **kwargs
    )
    
    # Show the plot
    show(fig, browser=None if open_browser else 'none')
    
    return fig


def plot_trailing_stop_analysis(
    df: pd.DataFrame,
    events: pd.DataFrame,
    trades_results: Dict[str, pd.DataFrame],
    filename: str = '',
    plot_width: int = None,
    show_legend: bool = True,
    open_browser: bool = True
):
    """
    Plot trailing stop analysis with multiple trailing percentages.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC(V) data. Must have columns: Open, High, Low, Close.
    events : pd.DataFrame
        DataFrame of events with index as timestamps and columns 'direction', 'pt', 'sl', 'entry_price'.
    trades_results : Dict[str, pd.DataFrame]
        Dictionary with trailing percentages as keys and trade results as values.
        Each trade result should have 'dynamic_stop_path' column if save_trail=True was used.
    filename : str
        If specified, the plot is saved to this file.
    plot_width : int
        Width of the plot in pixels.
    show_legend : bool
        Whether to show the legend.
    open_browser : bool
        Whether to open the browser when saving to file.
    
    Returns:
    --------
    fig : bokeh.plotting.figure
        The Bokeh figure object.
    """
    # Reset Bokeh state
    _bokeh_reset(filename)
    
    # Constants
    COLORS = [BEAR_COLOR, BULL_COLOR]
    BAR_WIDTH = .8
    NBSP = '\N{NBSP}' * 4
    
    # Check if we have a datetime index
    is_datetime_index = isinstance(df.index, pd.DatetimeIndex)
    
    # Ensure we have the required columns
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")
    
    # Set up plot dimensions
    if plot_width is None:
        plot_width = 1200
    plot_height = 600
    
    # Create figure
    fig = _figure(
        width=plot_width,
        height=plot_height,
        title="Trailing Stop Analysis",
        x_axis_type='datetime' if is_datetime_index else 'linear',
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll='wheel_zoom'
    )
    
    # Get the first event to determine the time range
    if events.empty:
        raise ValueError("Events DataFrame is empty")
    
    event_start = events.index[0]
    event_end = events['t1'].iloc[0] if 't1' in events.columns else df.index[-1]
    
    # Filter data to event time range
    event_data = df.loc[event_start:event_end]
    
    # Plot candlestick chart
    inc = event_data.Close >= event_data.Open
    dec = event_data.Close < event_data.Open
    
    # Convert datetime index to milliseconds for Bokeh
    if is_datetime_index:
        x_values = event_data.index.astype('int64') // 10**6  # Convert to milliseconds
    else:
        x_values = event_data.index
    
    # Plot candlesticks
    fig.segment(x_values, event_data.High, x_values, event_data.Low, color="black", line_width=1)
    
    # Bullish candles (green)
    fig.vbar(x_values[inc], BAR_WIDTH, event_data.Open[inc], event_data.Close[inc], 
             fill_color=BULL_COLOR, line_color="black", line_width=1)
    
    # Bearish candles (red)
    fig.vbar(x_values[dec], BAR_WIDTH, event_data.Open[dec], event_data.Close[dec], 
             fill_color=BEAR_COLOR, line_color="black", line_width=1)
    
    # Plot barriers
    pt_level = events['pt'].iloc[0]
    sl_level = events['sl'].iloc[0]
    
    # Take profit line
    pt_span = Span(location=pt_level, dimension='width', line_color='green', 
                   line_dash='dashed', line_width=2, line_alpha=0.7)
    fig.add_layout(pt_span)
    
    # Stop loss line
    sl_span = Span(location=sl_level, dimension='width', line_color='red', 
                   line_dash='dashed', line_width=2, line_alpha=0.7)
    fig.add_layout(sl_span)
    
    # Plot entry point
    entry_price = events['entry_price'].iloc[0]
    if is_datetime_index:
        entry_x = event_start.value // 10**6
    else:
        entry_x = event_start
    
    fig.triangle(entry_x, entry_price, size=15, color='blue', alpha=0.8)
    
    # Plot trailing stop paths
    colors = ['red', 'orange', 'purple', 'brown', 'pink']
    color_cycle = cycle(colors)
    
    for i, (pct, result) in enumerate(trades_results.items()):
        if 'dynamic_stop_path' in result.columns:
            trail_path = result['dynamic_stop_path'].iloc[0]
            
            if trail_path is not None and len(trail_path) > 0:
                # Convert trail path index to milliseconds if datetime
                if is_datetime_index:
                    trail_x = trail_path.index.astype('int64') // 10**6
                else:
                    trail_x = trail_path.index
                
                color = next(color_cycle)
                fig.line(trail_x, trail_path.values, 
                        line_width=2, line_color=color, line_alpha=0.8,
                        legend_label=f'Trailing {pct}%')
                
                # Plot exit point if exists
                if not pd.isna(result['exit_time'].iloc[0]):
                    exit_time = result['exit_time'].iloc[0]
                    exit_price = result['exit_price'].iloc[0]
                    
                    if is_datetime_index:
                        exit_x = exit_time.value // 10**6
                    else:
                        exit_x = exit_time
                    
                    fig.inverted_triangle(exit_x, exit_price, size=12, color=color, alpha=0.8)
    
    # Add legend items for barriers
    if show_legend:
        # Create invisible points for legend
        fig.line([0], [0], line_color='green', line_dash='dashed', line_width=2, 
                legend_label='Take Profit', visible=False)
        fig.line([0], [0], line_color='red', line_dash='dashed', line_width=2, 
                legend_label='Stop Loss', visible=False)
        fig.triangle([0], [0], size=15, color='blue', alpha=0.8, 
                    legend_label='Entry', visible=False)
        
        fig.legend.location = "top_left"
        fig.legend.click_policy = "hide"
    
    # Format axes
    if is_datetime_index:
        fig.xaxis.formatter = DatetimeTickFormatter(
            hours=["%H:%M"],
            days=["%m/%d"],
            months=["%m/%Y"],
            years=["%Y"]
        )
    
    fig.yaxis.formatter = NumeralTickFormatter(format='0.00')
    
    # Add grid
    fig.grid.grid_line_alpha = 0.3
    
    # Add watermark
    _watermark(fig)
    
    # Save or show
    if filename:
        filename = _windows_safe_filename(filename)
        output_file(filename)
    
    # Show the plot
    show(fig, browser=None if open_browser else 'none')
    
    return fig
