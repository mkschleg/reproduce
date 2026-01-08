"""
Plotting utilities for RL4PSJoint experiments.
"""

from .config import PlotConfig
from .styles import StyleRegistry
from .plotter import SensitivityPlotter
from .altair_plotter import AltairPlotter, ScaleRegistry, AltairPlotSpec
from .plotly_plotter import PlotlyPlotter, PlotlyPlotSpec
from .data_transforms import TransformSpec, aggregate_with_ci, apply_transform_spec
from .migrate import migrate_v1_to_v2

__all__ = [
    # Matplotlib
    "PlotConfig",
    "StyleRegistry",
    "SensitivityPlotter",
    # Altair
    "AltairPlotter",
    "ScaleRegistry",
    "AltairPlotSpec",
    # Plotly
    "PlotlyPlotter",
    "PlotlyPlotSpec",
    # Data transforms
    "TransformSpec",
    "aggregate_with_ci",
    "apply_transform_spec",
    # Migration
    "migrate_v1_to_v2",
]
