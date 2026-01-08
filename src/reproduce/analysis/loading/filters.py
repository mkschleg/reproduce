"""
Filtering utilities for experiment DataFrames.
"""

import pandas as pd
from typing import Optional, Any


class ExperimentFilter:
    """Utilities for filtering experiment data."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize filter with DataFrame.

        Args:
            df: DataFrame to filter
        """
        self.df = df

    def apply(
        self,
        query: Optional[str] = None,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Apply filters to DataFrame.

        Args:
            query: Pandas query string (e.g., "`seed` == 42")
            **kwargs: Column=value filters (e.g., seed=42, combo_agent__type="Centralized")
                     Note: Use double underscore (__) for dots in column names

        Returns:
            Filtered DataFrame

        Examples:
            >>> filter.apply(query="`combo_agent.type` == 'Centralized'")
            >>> filter.apply(seed=42)
            >>> filter.apply(combo_agent__type="Centralized", seed=42)
        """
        result = self.df

        # Apply query string
        if query:
            result = result.query(query)

        # Apply kwargs filters (convert __ to . for nested keys)
        for key, value in kwargs.items():
            # Replace __ with . for nested column names
            column_name = key.replace("__", ".")

            if column_name in result.columns:
                if isinstance(value, str):
                    result = result[result[column_name] == value]
                else:
                    result = result[result[column_name] == value]

        return result

    def by_hyperparameter(self, **kwargs: Any) -> pd.DataFrame:
        """
        Filter by hyperparameter values.

        Convenience method for filtering by hyperparameters.

        Args:
            **kwargs: Hyperparameter=value filters

        Returns:
            Filtered DataFrame

        Examples:
            >>> filter.by_hyperparameter(seed=42)
            >>> filter.by_hyperparameter(combo_agent__type="Centralized")
        """
        return self.apply(**kwargs)
