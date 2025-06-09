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
    def __init__(self, data, name=None, overlay=False, color=None, scatter=False, plot=True):
        self.data = np.asarray(data)
        self.name = name
        self.overlay = overlay
        self.color = color
        self.scatter = scatter
        self.plot = plot

def plot_market_data(
    df: pd.DataFrame,
    indicators: List[Indicator] = None,
    events: pd.DataFrame = None,
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
    
    # Check if we have a datetime index
    is_datetime_index = isinstance(df.index, pd.DatetimeIndex)
    
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
    
    # Set datetime formatter if we have datetime index
    if is_datetime_index:
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
    
    # Plot events
    def _plot_events():
        if events is None or events.empty:
            return None
            
        # Merge events with main dataframe to get the right indices
        event_indices = []
        event_y_values = []
        event_directions = []
        event_tooltips = []
        
        for event_time in events.index:
            # Find the corresponding index in the main dataframe
            if event_time in df.index:
                idx = df.index.get_loc(event_time)
            elif is_datetime_index and isinstance(event_time, pd.Timestamp):
                # Find nearest timestamp if exact match not found
                try:
                    idx = df.index.get_indexer([event_time], method='nearest')[0]
                except:
                    continue
            else:
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
    
    # Plot indicators
    def _plot_indicators():
        class LegendStr(str):
            # Ensures unique legend items even with same string content
            def __eq__(self, other):
                return self is other
        
        ohlc_colors = colorgen()
        indicator_figs = []
        
        for i, ind in enumerate(indicators):
            if not ind.plot:
                continue
                
            value = np.atleast_2d(ind.data)
            
            if ind.overlay:
                fig = fig_ohlc
            else:
                fig = new_indicator_figure()
                indicator_figs.append(fig)
                
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
            if len(value) == 1:
                fig.legend.glyph_width = 0
                
        return indicator_figs
    
    # Plot the components
    if plot_volume:
        fig_volume = _plot_volume_section()
        figs_below_ohlc.append(fig_volume)
        
    ohlc_bars = _plot_ohlc()
    
    # Plot events on the OHLC chart
    _plot_events()
    
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
