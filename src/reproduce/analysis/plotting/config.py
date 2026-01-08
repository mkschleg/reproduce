"""
Plot configuration for styling and behavior.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Dict, Tuple
import numpy as np


@dataclass
class PlotConfig:
    """Configuration for plot styling and behavior."""

    # Figure dimensions
    figsize: Tuple[int, int] = (15, 6)
    dpi: int = 100

    # Styling
    color_scheme: str = "tol_bright"  # "tol_bright", "tol_muted", "matplotlib"

    # Statistics
    default_agg_func: Callable = np.median
    default_ci_func: Optional[Callable] = None  # None = use default (25/75 quantiles)
    ci_alpha: float = 0.2

    # Labels
    simple_names: Optional[Dict[str, str]] = None  # Long names -> short names
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None

    # Legend
    show_legend: bool = True
    legend_location: str = "best"

    # Axes
    log_scale: bool = False
    ylims: Optional[Tuple[float, float]] = None
    xlims: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        """Set default CI function if not provided."""
        if self.default_ci_func is None:
            # Default: 25/75 quantiles
            self.default_ci_func = lambda x: (
                np.quantile(x, 0.25),
                np.quantile(x, 0.75)
            )

    @classmethod
    def defaults(cls) -> "PlotConfig":
        """Return default configuration."""
        return cls()

    @classmethod
    def for_agent_comparison(cls) -> "PlotConfig":
        """
        Pre-configured for comparing agent types.

        Returns:
            PlotConfig with sensible defaults for agent comparison
        """
        return cls(
            simple_names={
                "combo_agent.type": "Agent",
                "env.envs": "Env",
                "env.action_func.swap_prob": "$\\lambda$",
                "env.obs_func.kappa": "$\\kappa$",
                "base_dqn.q_network.width": "Width"
            }
        )

    @classmethod
    def for_publication(
        cls,
        figsize: Tuple[int, int] = (12, 8)
    ) -> "PlotConfig":
        """
        Pre-configured for publication-quality plots.

        Args:
            figsize: Figure size in inches

        Returns:
            PlotConfig with publication settings
        """
        return cls(
            figsize=figsize,
            dpi=300,
            simple_names={
                "combo_agent.type": "Agent",
                "env.envs": "Environment",
                "env.action_func.swap_prob": "$\\lambda$",
                "env.obs_func.kappa": "$\\kappa$"
            }
        )

    def get_simple_name(self, key: str) -> str:
        """
        Get simplified name for a key.

        Args:
            key: Full hyperparameter key

        Returns:
            Simplified name if available, otherwise original key
        """
        if self.simple_names is None:
            return key
        return self.simple_names.get(key, key)
