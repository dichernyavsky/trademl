"""
RegimeSearch: Decision tree for finding market regimes.

This package provides:
- Tree: Main decision tree class for regime detection
- Performance metrics: SharpePerf, SortinoPerf, KellyProxyPerf, etc.
- Stability metrics: UVxCoverageStability for reliability assessment
"""

from .tree import Tree
from .metrics import (
    SharpePerf,
    SortinoPerf, 
    KellyProxyPerf,
    UVxCoverageStability
)

__all__ = [
    'Tree',
    'SharpePerf',
    'SortinoPerf', 
    'KellyProxyPerf',
    'UVxCoverageStability'
]
