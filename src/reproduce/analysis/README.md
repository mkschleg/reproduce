# Analysis Module - DataLake Interface

A simplified interface for exploring RL experiment data with support for sensitivity curves, result exploration, and method comparison.

## Quick Start

```python
from analysis import DataLake

# Initialize and load experiment
lake = DataLake("datalake")
exp = lake.load("env_lambda_sweep_mes_10000")

# Create sensitivity plot
fig, axes = exp.plot_sensitivity(
    x_key="env.action_func.swap_prob",
    metric="returns:avg_end",
    group_by=["combo_agent.type"],
    facet_by="env.envs"
)
fig.savefig("sensitivity.pdf")
```

## Features

### DataLake
- **Auto-configured aggregates**: Default functions for avg_end, avg_all, median_end
- **List experiments**: `lake.list_experiments()`
- **Load experiments**: `lake.load("experiment_name")`
- **Load multiple**: `lake.load_multiple(["exp1", "exp2"])` - Load several at once
- **Custom aggregates**: `lake.register_agg_func(name, prep_func, agg_func)`

### Experiment
- **Summary**: `exp.summary()` - Get experiment statistics
- **Best hyperparameters**: `exp.best_hypers(sort_key, best_over)`
- **Filtering**: `exp.filter(query="...", **kwargs)`
- **Plotting**: `exp.plot_sensitivity(...)`, `exp.plot_individual_returns(...)`
- **DataFrame access**: `exp.df` - Direct access to underlying DataFrame
- **Combining**: `exp1.combine_with(exp2)` - Merge experiments for comparison
- **Multi-combine**: `Experiment.combine_multiple([exp1, exp2, ...])` - Merge many experiments

### Plotting
- **Sensitivity curves**: Line plots with error bars (metric vs hyperparameter)
- **Individual returns**: Per-agent return grids
- **Pre-configured styles**: tol_colors for agent types
- **Configurable CIs**: SEM or quantiles
- **PlotConfig**: Customize figure size, colors, labels, etc.

## Module Structure

```
analysis/
├── __init__.py          # Public API
├── notebook_config.py   # Shared notebook plotting + cleaning defaults
├── loading/             # Data-loading tools
│   ├── datalake.py       # DataLake class
│   ├── experiment.py     # Experiment wrapper
│   ├── aggregation.py    # Aggregate function registry
│   └── filters.py        # Query/filter utilities
└── plotting/            # Plotting tools
    ├── plotter.py        # SensitivityPlotter
    ├── config.py         # PlotConfig
    └── styles.py         # StyleRegistry
```

## Examples

### Custom Plot Configuration

```python
from analysis.plotting import PlotConfig

config = PlotConfig(
    figsize=(12, 8),
    log_scale=True,
    simple_names={"env.action_func.swap_prob": "$\\lambda$"},
    ylims=(-25000, -100)
)

fig, axes = exp.plot_sensitivity(
    x_key="env.action_func.swap_prob",
    config=config
)
```

### Notebook Defaults

```python
from analysis.notebook_config import notebook_plot_config, clean_experiment

config = notebook_plot_config()
exp2_cleaned = clean_experiment(exp2)
```

### Filtering

```python
# Filter with kwargs
centralized = exp.filter(combo_agent__type="Centralized")

# Filter with query string
high_lambda = exp.filter(query="`env.action_func.swap_prob` > 0.5")

# Combine filters
filtered = exp.filter(
    query="`env.action_func.swap_prob` > 0.5",
    combo_agent__type="Centralized"
)
```

### Best Hyperparameters

```python
# Find best over agent types and environments
best = exp.best_hypers(
    sort_key="returns:avg_end_mean",
    best_over=["combo_agent.type", "env.envs"]
)

# Export as sweep config
exp.export_sweep_config("configs/best_params.yaml")
```

### Individual Returns Plotting

```python
fig, axes = exp.plot_individual_returns(
    x_key="env.action_func.swap_prob",
    group_by=["combo_agent.type"],
    facet_by="env.envs",
    num_agents=2
)
```

### Multi-Experiment Comparison

```python
# Method 1: Load and combine two experiments
exp1 = lake.load("env_lambda_sweep_mes_10000")
exp2 = lake.load("env_lambda_sweep_dqn_dec_v2")

# Optional: Clean/filter before combining
exp2_filtered = exp2.filter(query="`env.max_episode_steps` == 10000")

# Combine
combined = exp1.combine_with(exp2_filtered, name="comparison")

# Method 2: Load multiple at once
exp_dict = lake.load_multiple([
    "env_lambda_sweep_mes_10000",
    "env_lambda_sweep_dqn_dec_v2"
])

# Combine all
from analysis import Experiment
combined = Experiment.combine_multiple(
    list(exp_dict.values()),
    name="multi_comparison"
)

# Plot from combined data
fig, axes = combined.plot_sensitivity(
    x_key="env.action_func.swap_prob",
    metric="returns:avg_end",
    group_by=["combo_agent.type"],
    facet_by="env.envs"
)
```

## Design Principles

1. **Interface-agnostic**: Core classes have no UI dependencies
2. **Backwards compatible**: Wraps existing `data_utils` functions
3. **Smart defaults**: Auto-detects common aggregates, pre-configured styling
4. **Full control**: Access to raw DataFrames and custom configurations
5. **Extensible**: Registries for custom aggregates and styles

## Future Extensions

- CLI interface for batch operations
- Web interface for shareable dashboards
- Cross-experiment comparison views
- Interactive widgets for marimo notebooks
