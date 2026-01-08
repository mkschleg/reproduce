"""
Aggregate function registry for experiment results.

Provides default aggregate functions (avg_end, avg_all, median_end)
and supports custom function registration.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Callable, Optional, List
import numpy as np
import rl4psjoint.utils.data_utils as data_utils
from rl4psjoint.utils.config import dataclass_factory


# ============================================================
# Config-based Aggregation Functions
# ============================================================

@dataclass_factory(key_name="type", strict=True, buildable=False)
class AggregationConfig:
    """
    Base class for aggregation function configurations.

    Subclasses register via type_value parameter and implement to_funcs().
    This allows YAML-based configuration of aggregation functions.
    """
    type: str
    name: Optional[str] = None  # Custom name for the aggregation (overrides type)

    def get_name(self) -> str:
        """Get the aggregation name (custom name or type)."""
        return self.name if self.name else self.type

    def to_funcs(self) -> Tuple[Callable, Callable]:
        """
        Convert config to (prep_func, agg_func) tuple.

        Returns:
            Tuple of (preparation_function, aggregation_function)
        """
        raise NotImplementedError


@dataclass
class LastNConfig(AggregationConfig, type_value="avg_end"):
    """Average/aggregation of last N episodes."""
    n: int = 100
    agg: str = "mean"  # "mean", "median", "std", "min", "max"

    def to_funcs(self) -> Tuple[Callable, Callable]:
        agg_func = self._get_numpy_agg(self.agg)
        return (
            data_utils.prep_include_last,
            data_utils.build_aggregate_func("agg_last_n", n=self.n, agg=agg_func)
        )

    @staticmethod
    def _get_numpy_agg(agg: str) -> Callable:
        """Map string to numpy aggregation function."""
        mapping = {
            "mean": np.mean,
            "median": np.median,
            "std": np.std,
            "min": np.min,
            "max": np.max,
        }
        if agg not in mapping:
            raise ValueError(f"Unknown aggregation: {agg}. Valid: {list(mapping.keys())}")
        return mapping[agg]


@dataclass
class MedianLastNConfig(AggregationConfig, type_value="median_end"):
    """Median of last N episodes."""
    n: int = 100

    def to_funcs(self) -> Tuple[Callable, Callable]:
        return (
            data_utils.prep_include_last,
            data_utils.build_aggregate_func("agg_last_n", n=self.n, agg=np.median)
        )


@dataclass
class AllEpisodesConfig(AggregationConfig, type_value="avg_all"):
    """Average/aggregation of all episodes."""
    agg: str = "mean"

    def to_funcs(self) -> Tuple[Callable, Callable]:
        agg_func = LastNConfig._get_numpy_agg(self.agg)
        return (data_utils.prep_include_last, agg_func)


@dataclass
class MedianAllConfig(AggregationConfig, type_value="median_all"):
    """Median of all episodes."""

    def to_funcs(self) -> Tuple[Callable, Callable]:
        return (data_utils.prep_include_last, np.median)


def aggregation_configs_to_dict(
    configs: List[AggregationConfig]
) -> Dict[str, Tuple[Callable, Callable]]:
    """
    Convert list of AggregationConfig objects to agg_funcs dict.

    Args:
        configs: List of aggregation configurations

    Returns:
        Dictionary mapping names to (prep_func, agg_func) tuples

    Example:
        >>> configs = [
        ...     AggregationConfig.from_config({"type": "avg_end", "n": 5, "name": "short"}),
        ...     AggregationConfig.from_config({"type": "avg_end", "n": 100, "name": "long"}),
        ... ]
        >>> agg_dict = aggregation_configs_to_dict(configs)
        >>> list(agg_dict.keys())
        ['short', 'long']
    """
    return {config.get_name(): config.to_funcs() for config in configs}


# ============================================================
# Legacy Registry-based Aggregation Functions
# ============================================================


class AggregationRegistry:
    """
    Registry for aggregate functions with auto-detection.

    Manages common aggregate patterns (avg_end, avg_all, median_end, etc.)
    """

    def __init__(self):
        self._functions: Dict[str, Tuple[Callable, Callable]] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register commonly used aggregate functions."""
        # Average of last 100 episodes (mean)
        self._functions["avg_end"] = (
            data_utils.prep_include_last,
            data_utils.build_aggregate_func("agg_last_n", n=100, agg=np.mean)
        )

        # Average of all episodes (mean)
        self._functions["avg_all"] = (
            data_utils.prep_include_last,
            np.mean
        )

        # Median of last 100 episodes
        self._functions["median_end"] = (
            data_utils.prep_include_last,
            data_utils.build_aggregate_func("agg_last_n", n=100, agg=np.median)
        )

        # Median of all episodes
        self._functions["median_all"] = (
            data_utils.prep_include_last,
            np.median
        )

    def get_defaults(self) -> Dict[str, Tuple[Callable, Callable]]:
        """Return default aggregate functions."""
        return self._functions.copy()

    def register(self, name: str, prep_func: Callable, agg_func: Callable):
        """
        Register custom aggregate function.

        Args:
            name: Name for the aggregate function
            prep_func: Preparation function (e.g., data_utils.prep_include_last)
            agg_func: Aggregation function (e.g., np.mean)
        """
        self._functions[name] = (prep_func, agg_func)

    def get(self, name: str) -> Tuple[Callable, Callable]:
        """
        Get aggregate function by name.

        Args:
            name: Name of aggregate function

        Returns:
            Tuple of (prep_func, agg_func)

        Raises:
            KeyError: If aggregate function not found
        """
        if name not in self._functions:
            raise KeyError(f"Aggregate function '{name}' not found. "
                          f"Available: {list(self._functions.keys())}")
        return self._functions[name]

    def list_available(self) -> list:
        """List all available aggregate function names."""
        return list(self._functions.keys())
