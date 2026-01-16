#!/usr/bin/env python3
"""
CLI tool for generating plots from YAML configs.

Usage:
    # Generate all plots to a directory
    uv run python plot_cli.py --config plt-notebooks/configs/example_altair_v2.yaml --output plots/ --format pdf

    # Generate a specific plot
    uv run python plot_cli.py --config plt-notebooks/configs/example_altair_v2.yaml --plot sensitivity_v2 --output my_plot.html

    # Generate all plots as PNG to current directory
    uv run python plot_cli.py --config plt-notebooks/configs/example_altair_v2.yaml --format png
"""

from pathlib import Path
from typing import Optional
import tyro
from dataclasses import dataclass

from reproduce_analysis.loading import parse_yaml_config, load_from_config
from reproduce_analysis.loading.yaml_utils import get_config_backend
from reproduce_analysis.plotting import PlotlyPlotter, AltairPlotter, ScaleRegistry


@dataclass
class PlotConfig:
    """Configuration for plot generation."""

    config: Path
    """Path to YAML config file."""

    plot: Optional[str] = None
    """Name of specific plot to generate. If not specified, generates all plots."""

    output: Optional[Path] = None
    """Output directory (for all plots) or file path (for single plot). If not specified, saves to current directory or '<plot_name>.<format>'."""

    format: str = "html"
    """Output format: html, png, pdf, svg, etc. Default: html"""

    backend: Optional[str] = None
    """Backend to use (plotly, altair). If not specified, uses backend from config."""

    show: bool = False
    """Open the plot in a browser after saving (only works for single plots with html format)."""


def generate_single_plot(
    plot_spec: dict,
    plot_name: str,
    loaded: dict,
    parsed: dict,
    backend: str,
    output_path: Path,
    show: bool = False,
) -> None:
    """Generate a single plot and save it."""
    print(f"📊 Generating plot: {plot_name}")

    # Get experiment data
    source = plot_spec.get("source", "combined")
    if source == "combined":
        exp = loaded.get("combined")
    else:
        exp = loaded.get("experiments", {}).get(source)

    if exp is None:
        raise ValueError(f"No experiment found for source: {source}")

    # Apply source_mode transformations if needed
    from analysis.loading import Experiment
    if plot_spec.get("source_mode") == "best_hypers":
        df = exp.best_hypers(
            sort_key=plot_spec.get("sort_key", "returns:avg_end_mean"),
            ascending=plot_spec.get("ascending", False),
            best_over=plot_spec.get("best_over", []),
        )
        exp = Experiment(df, f"{exp.name}_best", exp.metadata)

    # Apply filters
    filter_query = plot_spec.get("filter_query")
    filter_kwargs = plot_spec.get("filter_kwargs") or {}
    if filter_query or filter_kwargs:
        exp = exp.filter(query=filter_query, **filter_kwargs)

    # Create plotter and generate plot
    plotting = parsed.get("plotting", {})
    scales = ScaleRegistry(plotting.get("scales"))
    labels = plotting.get("labels", {})

    if backend == "plotly":
        plotter = PlotlyPlotter(exp, scales=scales, labels=labels)
        fig = plotter.from_yaml_spec(plot_spec)

        # Save
        print(f"💾 Saving to: {output_path}")
        if output_path.suffix == ".html":
            fig.write_html(output_path)
        else:
            fig.write_image(output_path)

        # Optionally show
        if show and output_path.suffix == ".html":
            import webbrowser
            webbrowser.open(output_path.absolute().as_uri())
            print(f"🌐 Opened in browser")

    elif backend == "altair":
        plotter = AltairPlotter(exp, scales=scales, labels=labels)
        chart = plotter.from_yaml_spec(plot_spec)

        # Save
        print(f"💾 Saving to: {output_path}")
        chart.save(str(output_path))

        # Optionally show
        if show and output_path.suffix == ".html":
            import webbrowser
            webbrowser.open(output_path.absolute().as_uri())
            print(f"🌐 Opened in browser")

    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'plotly' or 'altair'.")

    print(f"✅ Plot saved to {output_path}")


def main(config: PlotConfig) -> None:
    """Generate plot(s) from YAML configuration."""

    # Load and parse config
    print(f"📄 Loading config: {config.config}")
    with open(config.config, "r", encoding="utf-8") as f:
        yaml_text = f.read()

    parsed = parse_yaml_config(yaml_text)
    loaded = load_from_config(parsed)

    # Get plots from config
    plots = parsed.get("plots", [])
    if not plots:
        raise ValueError("No plots defined in config")

    # Determine backend
    backend = config.backend or get_config_backend(parsed)
    print(f"🎨 Using backend: {backend}")

    # Normalize format (remove leading dot if present)
    output_format = config.format.lstrip(".")

    if config.plot:
        # Generate single specific plot
        plot_spec = None
        for p in plots:
            if p.get("name") == config.plot:
                plot_spec = p
                break
        if not plot_spec:
            available = [p.get("name", f"plot_{i}") for i, p in enumerate(plots)]
            raise ValueError(
                f"Plot '{config.plot}' not found. Available plots: {', '.join(available)}"
            )
        plot_name = config.plot

        # Determine output path
        if config.output:
            output_path = config.output
            # If output has no extension, add the format
            if not output_path.suffix:
                output_path = output_path.with_suffix(f".{output_format}")
        else:
            output_path = Path(f"{plot_name}.{output_format}")

        generate_single_plot(
            plot_spec, plot_name, loaded, parsed, backend, output_path, config.show
        )

    else:
        # Generate all plots
        print(f"📊 Generating all {len(plots)} plots...")

        # Determine output directory
        if config.output:
            output_dir = Path(config.output)
        else:
            output_dir = Path(".")

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {output_dir}")

        # Generate each plot
        success_count = 0
        for i, plot_spec in enumerate(plots):
            plot_name = plot_spec.get("name", f"plot_{i}")
            output_path = output_dir / f"{plot_name}.{output_format}"

            try:
                generate_single_plot(
                    plot_spec, plot_name, loaded, parsed, backend, output_path, show=False
                )
                success_count += 1
            except Exception as e:
                print(f"❌ Error generating {plot_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n✅ Done! Generated {success_count}/{len(plots)} plots in {output_dir}")


if __name__ == "__main__":
    tyro.cli(main)
