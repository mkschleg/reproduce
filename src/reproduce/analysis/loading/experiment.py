"""
Experiment class - wrapper for experiment data with analysis methods.
"""

from typing import Optional, List, Dict, Any, TYPE_CHECKING
import pandas as pd
from ..utils import data_utils
from .filters import ExperimentFilter

if TYPE_CHECKING:
    from ..plotting.plotter import SensitivityPlotter
    from ..plotting.config import PlotConfig


class Experiment:
    """
    Represents a loaded experiment with analysis methods.

    Wraps DataFrame with metadata and provides high-level analysis operations.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        name: str,
        metadata: Optional[Dict] = None
    ):
        """
        Initialize experiment.

        Args:
            df: DataFrame containing experiment data
            name: Experiment name
            metadata: Optional metadata dictionary
        """
        self._df = df
        self.name = name
        self.metadata = metadata or {}
        self._filter = ExperimentFilter(df)
        self._plotter: Optional["SensitivityPlotter"] = None

    @property
    def df(self) -> pd.DataFrame:
        """Access underlying DataFrame."""
        return self._df

    @property
    def plotter(self) -> "SensitivityPlotter":
        """Lazy initialization of plotter."""
        if self._plotter is None:
            from ..plotting.plotter import SensitivityPlotter
            self._plotter = SensitivityPlotter(self)
        return self._plotter

    @property
    def hyperparams(self) -> List[str]:
        """
        List all hyperparameter columns.

        Returns:
            List of column names that are hyperparameters
        """
        result_cols = [c for c in self._df.columns
                      if "returns" in c or ":" in c or c.startswith("metric_")]
        return [c for c in self._df.columns
                if c not in result_cols + ["seed", "config_file"]]

    @property
    def metrics(self) -> List[str]:
        """
        List all metric columns (aggregated results).

        Returns:
            List of column names that are metrics
        """
        return [c for c in self._df.columns
                if "returns" in c or ":" in c or c.startswith("metric_")]

    def group_by(self, ignore_githash: bool = True) -> pd.DataFrame:
        """
        Group runs by hyperparameters.

        Wrapper around data_utils.get_group_by with experiment context.

        Args:
            ignore_githash: Whether to ignore githash in grouping

        Returns:
            Grouped DataFrame
        """
        return data_utils.get_group_by(self._df, ignore_githash=ignore_githash)

    def best_hypers(
        self,
        sort_key: str = "returns:avg_end_mean",
        ascending: bool = False,
        best_over: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Find best hyperparameters.

        Wrapper around data_utils.get_best_hypers with sensible defaults.

        Args:
            sort_key: Metric to sort by
            ascending: Sort order
            best_over: List of columns to get best over

        Returns:
            DataFrame with best hyperparameters
        """
        if best_over is None:
            best_over = []
        return data_utils.get_best_hypers(
            self._df,
            sort_key=sort_key,
            ascending=ascending,
            best_over=best_over
        )

    def filter(
        self,
        query: Optional[str] = None,
        **kwargs: Any
    ) -> "Experiment":
        """
        Filter experiment data.

        Args:
            query: Pandas query string
            **kwargs: Key-value filters (e.g., seed=42, combo_agent__type="Centralized")

        Returns:
            New Experiment with filtered data

        Examples:
            >>> filtered = exp.filter(query="`combo_agent.type` == 'Centralized'")
            >>> filtered = exp.filter(seed=42)
            >>> filtered = exp.filter(combo_agent__type="Centralized", seed=42)
        """
        filtered_df = self._filter.apply(query=query, **kwargs)
        return Experiment(filtered_df, f"{self.name}_filtered", self.metadata)

    def export_sweep_config(self, save_path: Optional[str] = None) -> Dict:
        """
        Export hyperparameters as sweep configuration.

        Wrapper around create_params_for_final_sweep.

        Args:
            save_path: Optional path to save YAML config

        Returns:
            Dictionary with base parameters and sweep configuration
        """
        return data_utils.create_params_for_final_sweep(self._df, save_file=save_path)

    def summary(self) -> Dict[str, Any]:
        """
        Return summary statistics about the experiment.

        Returns:
            Dictionary with experiment summary
        """
        return {
            "name": self.name,
            "num_runs": len(self._df),
            "num_configs": len(self.group_by()),
            "hyperparams": self.hyperparams,
            "metrics": self.metrics,
            "seeds": sorted(self._df["seed"].unique().tolist()) if "seed" in self._df else None,
            "metadata": self.metadata
        }

    # Plotting methods (delegate to SensitivityPlotter)
    def plot_sensitivity(
        self,
        x_key: str,
        metric: str = "returns:avg_end",
        group_by: Optional[List[str]] = None,
        facet_by: Optional[str] = None,
        config: Optional["PlotConfig"] = None,
        **kwargs
    ):
        """
        Create sensitivity curve plot.

        Args:
            x_key: Hyperparameter to plot on x-axis
            metric: Metric to plot on y-axis
            group_by: Keys to group/color series by
            facet_by: Key to create subplot facets
            config: PlotConfig for styling (None = defaults)
            **kwargs: Additional plot customization

        Returns:
            (fig, axes) tuple for further customization

        Examples:
            >>> fig, axes = exp.plot_sensitivity(
            ...     x_key="env.action_func.swap_prob",
            ...     metric="returns:avg_end",
            ...     group_by=["combo_agent.type"],
            ...     facet_by="env.envs"
            ... )
        """
        return self.plotter.sensitivity(
            x_key=x_key,
            metric=metric,
            group_by=group_by,
            facet_by=facet_by,
            config=config,
            **kwargs
        )

    def plot_individual_returns(
        self,
        x_key: str,
        group_by: Optional[List[str]] = None,
        facet_by: Optional[str] = None,
        num_agents: int = 2,
        metric_prefix: str = "sep_returns:end",
        config: Optional["PlotConfig"] = None,
        **kwargs
    ):
        """
        Create per-agent return plots.

        Args:
            x_key: Hyperparameter to plot on x-axis
            group_by: Keys to group/color series by
            facet_by: Key to create subplot facets
            num_agents: Number of agents (for grid layout)
            metric_prefix: Prefix for individual return metrics
            config: PlotConfig for styling (None = defaults)
            **kwargs: Additional plot customization

        Returns:
            (fig, axes) tuple for further customization
        """
        return self.plotter.individual_returns(
            x_key=x_key,
            group_by=group_by,
            facet_by=facet_by,
            num_agents=num_agents,
            metric_prefix=metric_prefix,
            config=config,
            **kwargs
        )

    def combine_with(
        self,
        other: "Experiment",
        name: Optional[str] = None
    ) -> "Experiment":
        """
        Combine this experiment with another experiment.

        Concatenates the DataFrames, useful for comparing across experiments.

        Args:
            other: Another Experiment to combine with
            name: Name for combined experiment (default: "name1+name2")

        Returns:
            New Experiment with combined data

        Examples:
            >>> exp1 = lake.load("env_lambda_sweep_mes_10000")
            >>> exp2 = lake.load("env_lambda_sweep_dqn_dec_v2")
            >>> combined = exp1.combine_with(exp2)
        """
        import pandas as pd

        combined_df = pd.concat([self._df, other._df], ignore_index=True)
        combined_name = name or f"{self.name}+{other.name}"

        combined_metadata = {
            "experiments": [self.name, other.name],
            "num_runs_per_exp": {
                self.name: len(self._df),
                other.name: len(other._df)
            }
        }

        return Experiment(combined_df, combined_name, combined_metadata)

    @staticmethod
    def combine_multiple(
        experiments: List["Experiment"],
        name: Optional[str] = None
    ) -> "Experiment":
        """
        Combine multiple experiments.

        Args:
            experiments: List of Experiments to combine
            name: Name for combined experiment

        Returns:
            New Experiment with combined data

        Examples:
            >>> exps = [lake.load(name) for name in exp_names]
            >>> combined = Experiment.combine_multiple(exps, "comparison")
        """
        import pandas as pd

        if len(experiments) == 0:
            raise ValueError("Must provide at least one experiment")

        combined_df = pd.concat([exp._df for exp in experiments], ignore_index=True)
        combined_name = name or "+".join(exp.name for exp in experiments)

        combined_metadata = {
            "experiments": [exp.name for exp in experiments],
            "num_runs_per_exp": {
                exp.name: len(exp._df) for exp in experiments
            }
        }

        return Experiment(combined_df, combined_name, combined_metadata)

    def __repr__(self) -> str:
        """String representation."""
        return (f"Experiment(name='{self.name}', "
                f"num_runs={len(self._df)}, "
                f"num_configs={len(self.group_by())})")
