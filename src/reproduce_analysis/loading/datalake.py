"""
DataLake class - main entry point for experiment exploration.
"""

from pathlib import Path
from typing import Optional, Dict, List
import rl4psjoint.utils.data_utils as data_utils
from .experiment import Experiment
from .aggregation import AggregationRegistry


class DataLake:
    """
    Main interface for exploring RL experiment data.

    Usage:
        lake = DataLake("datalake")
        exp = lake.load("env_lambda_sweep_mes_10000")
        fig, axes = exp.plot_sensitivity(x_key="env.action_func.swap_prob")
    """

    def __init__(self, base_path: str = "datalake"):
        """
        Initialize DataLake with base directory path.

        Args:
            base_path: Path to datalake directory containing experiments
        """
        self.base_path = Path(base_path)
        if not self.base_path.exists():
            raise ValueError(f"DataLake path does not exist: {self.base_path}")
        self._registry = AggregationRegistry()

    def list_experiments(self) -> List[str]:
        """
        List all available experiments in the datalake.

        Returns:
            List of experiment directory names
        """
        if not self.base_path.is_dir():
            return []

        experiments = []
        for item in self.base_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                experiments.append(item.name)

        return sorted(experiments)

    def load(
        self,
        experiment_name: str,
        agg_funcs: Optional[Dict] = None,
        include_res: bool = False,
        progbar: bool = False
    ) -> Experiment:
        """
        Load a single experiment by name.

        Args:
            experiment_name: Name of experiment directory
            agg_funcs: Custom aggregate functions (None = auto-detect defaults)
            include_res: Include raw results arrays in DataFrame
            progbar: Show progress bar during loading

        Returns:
            Experiment object wrapping the loaded data

        Examples:
            >>> lake = DataLake("datalake")
            >>> exp = lake.load("env_lambda_sweep_mes_10000")
            >>> print(exp.summary())
        """
        exp_path = self.base_path / experiment_name

        if not exp_path.exists():
            available = self.list_experiments()
            raise ValueError(
                f"Experiment '{experiment_name}' not found. "
                f"Available experiments: {available}"
            )

        # Use default aggregates if none provided
        if agg_funcs is None:
            agg_funcs = self._registry.get_defaults()

        # Load experiment using data_utils
        df = data_utils.load_exp_dir(
            str(exp_path),
            agg_funcs,
            include_res=include_res,
            progbar=progbar
        )

        # Create metadata
        metadata = {
            "path": str(exp_path),
            "num_runs": len(df),
            "agg_funcs": list(agg_funcs.keys())
        }

        return Experiment(df, experiment_name, metadata)

    def load_multiple(
        self,
        experiment_names: List[str],
        agg_funcs: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, Experiment]:
        """
        Load multiple experiments and return as dict.

        Args:
            experiment_names: List of experiment directory names
            agg_funcs: Custom aggregate functions (None = auto-detect)
            **kwargs: Additional arguments passed to load()

        Returns:
            Dictionary mapping experiment names to Experiment objects

        Examples:
            >>> lake = DataLake()
            >>> experiments = lake.load_multiple([
            ...     "env_lambda_sweep_mes_10000",
            ...     "env_lambda_sweep_dqn_dec_v2"
            ... ])
        """
        experiments = {}
        for name in experiment_names:
            experiments[name] = self.load(name, agg_funcs=agg_funcs, **kwargs)
        return experiments

    @property
    def default_agg_funcs(self) -> Dict:
        """Return default aggregate functions."""
        return self._registry.get_defaults()

    def register_agg_func(self, name: str, prep_func, agg_func):
        """
        Register a custom aggregate function.

        Args:
            name: Name for the aggregate function
            prep_func: Preparation function
            agg_func: Aggregation function

        Examples:
            >>> import numpy as np
            >>> lake.register_agg_func(
            ...     "avg_first_10",
            ...     lambda row: ("returns", row["returns"]),
            ...     lambda x: np.mean(x[:10])
            ... )
        """
        self._registry.register(name, prep_func, agg_func)
