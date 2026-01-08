"""
SensitivityPlotter - reusable plotting utilities for RL experiments.
"""

from typing import Optional, List, Tuple, Callable, Any, TYPE_CHECKING
import matplotlib.pyplot as plt
import matplotlib.figure as mfig
import numpy as np
import itertools
from ..utils import data_utils
from .config import PlotConfig
from .styles import StyleRegistry

if TYPE_CHECKING:
    from ..experiment import Experiment


class SensitivityPlotter:
    """
    Reusable plotting utilities for RL experiments.

    Pre-configured with common color maps, CI methods, and plot layouts.
    """

    def __init__(self, experiment: "Experiment"):
        """
        Initialize plotter with experiment.

        Args:
            experiment: Experiment object to plot
        """
        self.experiment = experiment
        self.styles = StyleRegistry()

    def sensitivity(
        self,
        x_key: str,
        metric: str = "returns:avg_end",
        group_by: Optional[List[str]] = None,
        facet_by: Optional[str] = None,
        agg_func: Optional[Callable] = None,
        ci_func: Optional[Callable] = None,
        config: Optional[PlotConfig] = None,
        **kwargs
    ) -> Tuple[mfig.Figure, np.ndarray]:
        """
        Create sensitivity curve plot (metric vs hyperparameter).

        Args:
            x_key: Hyperparameter for x-axis
            metric: Metric for y-axis
            group_by: Keys to group series by (different lines)
            facet_by: Key to create subplot facets
            agg_func: Aggregation function (default: from config or median)
            ci_func: Confidence interval function (default: from config or 25/75 quantiles)
            config: Plot styling configuration
            **kwargs: Additional matplotlib kwargs

        Returns:
            (fig, axes) tuple

        Examples:
            >>> fig, axes = plotter.sensitivity(
            ...     x_key="env.action_func.swap_prob",
            ...     metric="returns:avg_end",
            ...     group_by=["combo_agent.type"],
            ...     facet_by="env.envs"
            ... )
        """
        config = config or PlotConfig.defaults()
        agg_func = agg_func or config.default_agg_func
        ci_func = ci_func or config.default_ci_func

        # Get grouped DataFrame
        group_df = data_utils.get_group_by(self.experiment.df, ignore_githash=True)
        group_df = group_df.sort_values(x_key)

        # Determine faceting structure
        if facet_by is not None:
            facet_values = sorted(group_df[facet_by].unique())
            num_facets = len(facet_values)
            fig, axes = plt.subplots(1, num_facets, figsize=config.figsize, dpi=config.dpi)
            if num_facets == 1:
                axes = np.array([axes])
        else:
            fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
            axes = np.array([ax])
            facet_values = [None]

        # Plot each facet
        for idx, facet_value in enumerate(facet_values):
            cur_ax = axes[idx] if len(axes) > 1 else axes[0]

            # Filter to facet
            if facet_by is not None:
                facet_df = group_df[group_df[facet_by] == facet_value]
                cur_ax.set_title(f"{config.get_simple_name(facet_by)}: {facet_value}")
            else:
                facet_df = group_df

            # Determine grouping keys
            if group_by is not None:
                group_keys = group_by
            else:
                group_keys = []

            # Plot each group
            if len(group_keys) > 0:
                # Get all combinations of group values
                group_args = {k: np.sort(facet_df[k].unique()) for k in group_keys}
                prod_params = itertools.product(*[group_args[k] for k in group_keys])

                for params in prod_params:
                    # Build query
                    query = " and ".join([
                        f"`{k}` == '{v}'" if isinstance(v, str) else f"`{k}` == {v}"
                        for k, v in zip(group_keys, params)
                    ])
                    sub_df = facet_df.query(query)

                    if len(sub_df) == 0:
                        continue

                    # Extract data
                    raw_res = sub_df[metric]
                    x = sub_df[x_key]

                    # Aggregate
                    res = np.array([agg_func(v) for v in raw_res.tolist()])
                    cis = np.array([ci_func(v) for v in raw_res.tolist()])

                    # Get style kwargs
                    value_key_pairs = list(zip(params, group_keys))
                    label = ", ".join(
                        f"{config.get_simple_name(k)}: {v}"
                        for v, k in value_key_pairs
                    )
                    style_kwargs = self.styles.get_kwargs(value_key_pairs)
                    style_kwargs.update(kwargs)

                    # Plot line
                    cur_ax.plot(x, res, label=label, **style_kwargs)

                    # Plot confidence interval
                    if len(cis.shape) == 1:
                        # Symmetric CI
                        cur_ax.fill_between(
                            x, res - cis, res + cis,
                            alpha=config.ci_alpha,
                            color=style_kwargs.get("color", None)
                        )
                    else:
                        # Asymmetric CI (quantiles)
                        cur_ax.fill_between(
                            x, cis[:, 0], cis[:, 1],
                            alpha=config.ci_alpha,
                            color=style_kwargs.get("color", None)
                        )
            else:
                # No grouping - single line
                raw_res = facet_df[metric]
                x = facet_df[x_key]
                res = np.array([agg_func(v) for v in raw_res.tolist()])
                cis = np.array([ci_func(v) for v in raw_res.tolist()])

                cur_ax.plot(x, res, **kwargs)

                if len(cis.shape) == 1:
                    cur_ax.fill_between(
                        x, res - cis, res + cis,
                        alpha=config.ci_alpha
                    )
                else:
                    cur_ax.fill_between(
                        x, cis[:, 0], cis[:, 1],
                        alpha=config.ci_alpha
                    )

            # Apply styling
            if config.show_legend and len(group_keys) > 0:
                cur_ax.legend(loc=config.legend_location)

            if config.log_scale:
                cur_ax.set_yscale("symlog")

            if config.ylims is not None:
                cur_ax.set_ylim(config.ylims)

            if config.xlims is not None:
                cur_ax.set_xlim(config.xlims)

            # Set labels
            if config.xlabel is not None:
                cur_ax.set_xlabel(config.xlabel)
            else:
                cur_ax.set_xlabel(config.get_simple_name(x_key))

            if config.ylabel is not None:
                cur_ax.set_ylabel(config.ylabel)
            else:
                cur_ax.set_ylabel(config.get_simple_name(metric))

        plt.tight_layout()
        return fig, axes

    def individual_returns(
        self,
        x_key: str,
        group_by: Optional[List[str]] = None,
        facet_by: Optional[str] = None,
        num_agents: int = 2,
        metric_prefix: str = "sep_returns:end",
        agg_func: Optional[Callable] = None,
        ci_func: Optional[Callable] = None,
        config: Optional[PlotConfig] = None,
        prep_ind_returns: Optional[Callable] = None,
        **kwargs
    ) -> Tuple[mfig.Figure, np.ndarray]:
        """
        Create per-agent return sensitivity plots.

        Creates grid of subplots with one row per agent.

        Args:
            x_key: Hyperparameter for x-axis
            group_by: Keys to group series by
            facet_by: Key to create subplot facets (columns)
            num_agents: Number of agents (rows in grid)
            metric_prefix: Prefix for individual return metrics
            agg_func: Aggregation function
            ci_func: Confidence interval function
            config: Plot styling configuration
            prep_ind_returns: Function to extract individual returns from metric
            **kwargs: Additional matplotlib kwargs

        Returns:
            (fig, axes) tuple with shape (num_agents, num_facets)
        """
        config = config or PlotConfig.defaults()
        agg_func = agg_func or config.default_agg_func
        ci_func = ci_func or config.default_ci_func

        # Default prep function: extract agent i returns
        if prep_ind_returns is None:
            prep_ind_returns = lambda x, i: [v[i] for v in x]

        # Get grouped DataFrame
        group_df = data_utils.get_group_by(self.experiment.df, ignore_githash=True)
        group_df = group_df.sort_values(x_key)

        # Determine faceting structure
        if facet_by is not None:
            facet_values = sorted(group_df[facet_by].unique())
            num_facets = len(facet_values)
        else:
            facet_values = [None]
            num_facets = 1

        # Create subplot grid (num_agents rows, num_facets columns)
        fig, axes = plt.subplots(
            num_agents, num_facets,
            figsize=(config.figsize[0], config.figsize[1] * num_agents / 2),
            dpi=config.dpi
        )

        # Ensure axes is 2D
        if num_agents == 1 and num_facets == 1:
            axes = np.array([[axes]])
        elif num_agents == 1:
            axes = axes.reshape(1, -1)
        elif num_facets == 1:
            axes = axes.reshape(-1, 1)

        # Determine grouping keys
        if group_by is not None:
            group_keys = group_by
        else:
            group_keys = []

        # Plot each agent
        for agent_idx in range(num_agents):
            # Plot each facet
            for facet_idx, facet_value in enumerate(facet_values):
                cur_ax = axes[agent_idx, facet_idx]

                # Filter to facet
                if facet_by is not None:
                    facet_df = group_df[group_df[facet_by] == facet_value]
                    if agent_idx == 0:  # Only set title for top row
                        cur_ax.set_title(f"{config.get_simple_name(facet_by)}: {facet_value}")
                else:
                    facet_df = group_df

                # Plot each group
                if len(group_keys) > 0:
                    group_args = {k: np.sort(facet_df[k].unique()) for k in group_keys}
                    prod_params = itertools.product(*[group_args[k] for k in group_keys])

                    for params in prod_params:
                        # Build query
                        query = " and ".join([
                            f"`{k}` == '{v}'" if isinstance(v, str) else f"`{k}` == {v}"
                            for k, v in zip(group_keys, params)
                        ])
                        sub_df = facet_df.query(query)

                        if len(sub_df) == 0:
                            continue

                        # Extract individual returns for this agent
                        raw_res = [
                            prep_ind_returns(v, agent_idx)
                            for v in sub_df[metric_prefix].tolist()
                        ]
                        x = sub_df[x_key]

                        # Aggregate
                        res = np.array([agg_func(v) for v in raw_res])
                        cis = np.array([ci_func(v) for v in raw_res])

                        # Get style kwargs
                        value_key_pairs = list(zip(params, group_keys))
                        label = ", ".join(
                            f"{config.get_simple_name(k)}: {v}"
                            for v, k in value_key_pairs
                        )
                        style_kwargs = self.styles.get_kwargs(value_key_pairs)
                        style_kwargs.update(kwargs)

                        # Plot line
                        cur_ax.plot(x, res, label=label, **style_kwargs)

                        # Plot confidence interval
                        if len(cis.shape) == 1:
                            cur_ax.fill_between(
                                x, res - cis, res + cis,
                                alpha=config.ci_alpha,
                                color=style_kwargs.get("color", None)
                            )
                        else:
                            cur_ax.fill_between(
                                x, cis[:, 0], cis[:, 1],
                                alpha=config.ci_alpha,
                                color=style_kwargs.get("color", None)
                            )

                # Apply styling
                if config.show_legend and len(group_keys) > 0 and agent_idx == 0 and facet_idx == 0:
                    cur_ax.legend(loc=config.legend_location)

                if config.log_scale:
                    cur_ax.set_yscale("symlog")

                if config.ylims is not None:
                    cur_ax.set_ylim(config.ylims)

                if config.xlims is not None:
                    cur_ax.set_xlim(config.xlims)

                # Set labels
                if agent_idx == num_agents - 1:  # Bottom row
                    if config.xlabel is not None:
                        cur_ax.set_xlabel(config.xlabel)
                    else:
                        cur_ax.set_xlabel(config.get_simple_name(x_key))

                if facet_idx == 0:  # Left column
                    cur_ax.set_ylabel(f"Agent {agent_idx} Return")

        plt.tight_layout()
        return fig, axes
