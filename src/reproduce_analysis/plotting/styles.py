"""
Style registry for consistent plotting across experiments.
"""

from typing import Dict, Any, List, Tuple, Optional
import tol_colors


class StyleRegistry:
    """
    Registry of plotting styles for common agent types and configurations.

    Extensible via register() method.
    """

    def __init__(self):
        """Initialize with default agent styles."""
        self._agent_styles = self._default_agent_styles()

    def _default_agent_styles(self) -> Dict[str, Dict[str, Any]]:
        """Default styles for common agent types."""
        return {
            "Centralized": {
                "color": tol_colors.bright[0],
                "linestyle": "solid",
                "marker": "o"
            },
            "Decentralized": {
                "color": tol_colors.bright[1],
                "linestyle": "dashed",
                "marker": "s"
            },
            "DecentralizedJointObs": {
                "color": tol_colors.bright[1],
                "linestyle": "solid",
                "marker": "^"
            },
            "DecentralizedSumRew": {
                "color": tol_colors.bright[2],
                "linestyle": "dashed",
                "marker": "v"
            },
            "DecentralizedJointObsSumRew": {
                "color": tol_colors.bright[2],
                "linestyle": "solid",
                "marker": "d"
            }
        }

    def get(self, agent_type: str, key: Optional[str] = None) -> Any:
        """
        Get style for agent type.

        Args:
            agent_type: Agent type name
            key: Specific style attribute (color, linestyle, marker) or None for all

        Returns:
            Style value or dict of all styles
        """
        style = self._agent_styles.get(agent_type, {})
        return style.get(key) if key else style

    def register(self, agent_type: str, style: Dict[str, Any]):
        """
        Register custom style for agent type.

        Args:
            agent_type: Agent type name
            style: Dictionary of matplotlib style kwargs
        """
        self._agent_styles[agent_type] = style

    def get_kwargs(
        self,
        value_key_pairs: List[Tuple[Any, str]]
    ) -> Dict[str, Any]:
        """
        Extract matplotlib kwargs from hyperparameter value-key pairs.

        Args:
            value_key_pairs: List of (value, key) tuples from hyperparameters

        Returns:
            Dict of matplotlib kwargs (color, linestyle, marker)

        Examples:
            >>> registry = StyleRegistry()
            >>> pairs = [("Centralized", "combo_agent.type")]
            >>> kwargs = registry.get_kwargs(pairs)
            >>> # Returns: {"color": ..., "linestyle": "solid", "marker": "o"}
        """
        # Find agent type in pairs
        agent_type = None
        for value, key in value_key_pairs:
            if key == "combo_agent.type":
                agent_type = value
                break

        if agent_type is not None:
            return self.get(agent_type) or {}
        return {}

    def list_registered(self) -> List[str]:
        """List all registered agent types."""
        return list(self._agent_styles.keys())
