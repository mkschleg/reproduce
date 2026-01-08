"""
Plotly-based plotter for RL experiments.

This module re-exports from the refactored plotly submodule for backward compatibility.
"""

# Re-export from new location
from .plotly import PlotlyPlotter, PlotlyPlotSpec

__all__ = ["PlotlyPlotter", "PlotlyPlotSpec"]
