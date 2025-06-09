"""
Market Data Plotting Module

This module provides tools for visualizing market data and technical indicators
using Bokeh for interactive plots.
"""

from .plot import (
    plot_market_data,
    Indicator,
    set_bokeh_output,
    colorgen,
    lightness
)

# Set default output to notebook if running in Jupyter
import os
if 'JPY_PARENT_PID' in os.environ or 'inline' in os.environ.get('MPLBACKEND', ''):
    set_bokeh_output(notebook=True)

__all__ = [
    'plot_market_data',
    'Indicator',
    'set_bokeh_output',
    'colorgen',
    'lightness'
] 