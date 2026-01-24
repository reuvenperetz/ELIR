#!/usr/bin/env python3
"""
Script to plot NFE vs Latency graphs for different models and input sizes.

Usage:
    python plot_nfe_latency.py
    python plot_nfe_latency.py --output figures/nfe_latency_comparison.png
    python plot_nfe_latency.py --show  # Display interactive plot

Data format:
    Each model has data as list of tuples: (input_size, NFE, latency_ms)
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

# =============================================================================
# DATA DEFINITION
# Define your measurements here as (input_size, NFE, latency_ms, macs_g)
# macs_g = MACs in GMACs (billions of multiply-accumulate operations)
# =============================================================================

x = 0

# Face Restoration model data
FACE_RESTORATION_DATA = [
    # (input_size, NFE, latency_ms, macs_g)
    (256, 1, 12.69, 21.37),
    (256, 2, 16.20, 23.43),
    (256, 4, 24.03, 27.56),
    (256, 8, 37.74, 35.81),
    (512, 1, 33.54, 85.46),
    (512, 2, 40.66, 93.72),
    (512, 4, 54.52, 110.22),
    (512, 8, 82.54, 143.23),
]

# Blind Super Resolution model data
BLIND_SR_DATA = [
    # (input_size, NFE, latency_ms, macs_g)
    (256, 1, 12.34, 20.47),
    (256, 2, 16.29, 21.63),
    (256, 4, 21.92, 23.95),
    (256, 8, 34.47, 28.60),
    (512, 1, 31.72, 81.86),
    (512, 2, 37.21, 86.51),
    (512, 4, 47.63, 95.80),
    (512, 8, 69.82, 114.39),
]

# Reddit model data (example - replace with actual measurements)
REDDIT_DATA = [
    # (input_size, NFE, latency_ms, macs_g)
    (256, 1, 264.36, 33.82),
    (256, 2, 531.62, 67.64),
    (256, 4, 1053.53, 135.28),
    (256, 8, 2098.62, 270.54),
    (512, 1, 413.61, 134.46),
    (512, 2, 832.97, 268.93),
    (512, 4, 1637.44, 537.85),
    (512, 8, 3417.37, 1075.70),
]

# Model configurations
MODELS = {
    'Face Restoration': {
        'data': FACE_RESTORATION_DATA,
        'color': '#2ecc71',  # Green
        'marker': 'o',
        'params_m': 26.89,  # Number of parameters in millions
    },
    'Blind SR': {
        'data': BLIND_SR_DATA,
        'color': '#3498db',  # Blue
        'marker': 's',
        'params_m': 19.08,  # Number of parameters in millions
    },
    'Reddit': {
        'data': REDDIT_DATA,
        'color': '#e74c3c',  # Red
        'marker': '^',
        'params_m': 17.43,  # Number of parameters in millions
    },
}


def organize_data_by_input_size(models: Dict) -> Dict[int, Dict[str, List[Tuple[int, float, float]]]]:
    """
    Organize data by input size for easier plotting.

    Returns:
        Dict mapping input_size -> {model_name: [(NFE, latency, macs), ...]}
    """
    organized = defaultdict(lambda: defaultdict(list))

    for model_name, model_info in models.items():
        for input_size, nfe, latency, macs in model_info['data']:
            organized[input_size][model_name].append((nfe, latency, macs))

    # Sort by NFE within each model
    for input_size in organized:
        for model_name in organized[input_size]:
            organized[input_size][model_name].sort(key=lambda x: x[0])

    return dict(organized)


def plot_nfe_latency_per_input_size(models: Dict, output_path: str = None, show: bool = False):
    """
    Create one subplot per input size, showing NFE vs Latency for all models.
    """
    organized_data = organize_data_by_input_size(models)
    input_sizes = sorted(organized_data.keys())
    n_plots = len(input_sizes)

    if n_plots == 0:
        print("No data to plot!")
        return

    # Create figure with subplots (2 rows: latency and MACs)
    fig, axes = plt.subplots(2, n_plots, figsize=(5 * n_plots, 10))
    if n_plots == 1:
        axes = axes.reshape(2, 1)

    # Row 0: Latency plots
    for col, input_size in enumerate(input_sizes):
        ax = axes[0, col]
        for model_name, model_info in models.items():
            data = organized_data[input_size].get(model_name, [])
            if not data:
                continue

            nfes = [d[0] for d in data]
            latencies = [d[1] for d in data]
            params_m = model_info.get('params_m', 0)
            label = f"{model_name} ({params_m:.1f}M params)"

            ax.plot(nfes, latencies,
                    color=model_info['color'],
                    marker=model_info['marker'],
                    markersize=8,
                    linewidth=2,
                    label=label)

        ax.set_xlabel('NFE (Number of Function Evaluations)', fontsize=11)
        ax.set_ylabel('Latency (ms)', fontsize=11)
        ax.set_title(f'Input Size: {input_size}×{input_size}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    # Row 1: MACs plots
    for col, input_size in enumerate(input_sizes):
        ax = axes[1, col]
        for model_name, model_info in models.items():
            data = organized_data[input_size].get(model_name, [])
            if not data:
                continue

            nfes = [d[0] for d in data]
            macs = [d[2] for d in data]
            params_m = model_info.get('params_m', 0)
            label = f"{model_name} ({params_m:.1f}M params)"

            ax.plot(nfes, macs,
                    color=model_info['color'],
                    marker=model_info['marker'],
                    markersize=8,
                    linewidth=2,
                    label=label)

        ax.set_xlabel('NFE (Number of Function Evaluations)', fontsize=11)
        ax.set_ylabel('MACs (GMACs)', fontsize=11)
        ax.set_title(f'Input Size: {input_size}×{input_size}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    plt.suptitle('NFE vs Latency Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {output_path}")

    if show:
        plt.show()

    plt.close()


def plot_combined_view(models: Dict, output_path: str = None, show: bool = False):
    """
    Create a combined view with NFE vs Latency (log scale) for all input sizes.
    """
    organized_data = organize_data_by_input_size(models)
    input_sizes = sorted(organized_data.keys())

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    # Plot: NFE vs Latency with different line styles per input size
    line_styles = ['-', '--', '-.', ':']

    for model_name, model_info in models.items():
        for idx, input_size in enumerate(input_sizes):
            data = organized_data[input_size].get(model_name, [])
            if not data:
                continue

            nfes = [d[0] for d in data]
            latencies = [d[1] for d in data]

            style = line_styles[idx % len(line_styles)]
            label = f"{model_name} ({input_size}px)"

            ax1.plot(nfes, latencies,
                    color=model_info['color'],
                    marker=model_info['marker'],
                    markersize=6,
                    linewidth=2,
                    linestyle=style,
                    label=label,
                    alpha=0.8)

    ax1.set_xlabel('NFE', fontsize=11)
    ax1.set_ylabel('Latency (ms)', fontsize=11)
    ax1.set_title('NFE vs Latency (All Input Sizes)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')


    plt.tight_layout()

    if output_path:
        base_path = output_path.rsplit('.', 1)[0]
        combined_path = f"{base_path}_combined.png"
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        print(f"Saved combined plot to: {combined_path}")

    if show:
        plt.show()

    plt.close()


def plot_bar_comparison(models: Dict, output_path: str = None, show: bool = False):
    """
    Create bar chart comparing latency across models for each input size and NFE.
    """
    organized_data = organize_data_by_input_size(models)
    input_sizes = sorted(organized_data.keys())
    model_names = list(models.keys())
    n_models = len(model_names)

    # Get all unique NFEs
    all_nfes = set()
    for model_info in models.values():
        for _, nfe, _, _ in model_info['data']:
            all_nfes.add(nfe)
    nfes = sorted(all_nfes)

    n_input_sizes = len(input_sizes)
    fig, axes = plt.subplots(1, n_input_sizes, figsize=(6 * n_input_sizes, 5))
    if n_input_sizes == 1:
        axes = [axes]

    bar_width = 0.8 / n_models

    for ax, input_size in zip(axes, input_sizes):
        x = np.arange(len(nfes))

        for i, model_name in enumerate(model_names):
            data_dict = {nfe: lat for nfe, lat, macs in organized_data[input_size].get(model_name, [])}
            latencies = [data_dict.get(nfe, 0) for nfe in nfes]

            offset = (i - n_models / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, latencies, bar_width,
                         label=model_name,
                         color=models[model_name]['color'],
                         alpha=0.8)

            # Add value labels on bars
            for bar, lat in zip(bars, latencies):
                if lat > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{lat:.0f}', ha='center', va='bottom', fontsize=7)

        ax.set_xlabel('NFE', fontsize=11)
        ax.set_ylabel('Latency (ms)', fontsize=11)
        ax.set_title(f'Input Size: {input_size}×{input_size}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(nfes)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Latency Comparison by Model, NFE, and Input Size', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        base_path = output_path.rsplit('.', 1)[0]
        bar_path = f"{base_path}_bars.png"
        plt.savefig(bar_path, dpi=150, bbox_inches='tight')
        print(f"Saved bar plot to: {bar_path}")

    if show:
        plt.show()

    plt.close()


def print_data_table(models: Dict):
    """Print a summary table of all data."""
    organized_data = organize_data_by_input_size(models)
    input_sizes = sorted(organized_data.keys())
    model_names = list(models.keys())

    print("\n" + "=" * 90)
    print("MODEL PARAMETERS")
    print("=" * 90)
    for model_name in model_names:
        params_m = models[model_name].get('params_m', 0)
        print(f"{model_name:<20}: {params_m:.2f}M parameters")

    print("\n" + "=" * 90)
    print("DATA SUMMARY")
    print("=" * 90)

    for input_size in input_sizes:
        print(f"\nInput Size: {input_size}×{input_size}")
        print("-" * 80)
        print(f"{'Model':<20} {'NFE':<8} {'Latency (ms)':<15} {'MACs (G)':<12} {'FPS':<10}")
        print("-" * 80)

        for model_name in model_names:
            data = organized_data[input_size].get(model_name, [])
            for nfe, latency, macs in data:
                fps = 1000.0 / latency if latency > 0 else 0
                print(f"{model_name:<20} {nfe:<8} {latency:<15.2f} {macs:<12.2f} {fps:<10.2f}")


def main():
    parser = argparse.ArgumentParser(description='Plot NFE vs Latency graphs')
    parser.add_argument('--output', '-o', type=str, default='figures/nfe_latency.png',
                        help='Output path for the plot (default: figures/nfe_latency.png)')
    parser.add_argument('--show', action='store_true',
                        help='Display interactive plot')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save the plots')
    parser.add_argument('--plot-type', type=str, default='all',
                        choices=['line', 'combined', 'bar', 'all'],
                        help='Type of plot to generate')

    args = parser.parse_args()

    output_path = None if args.no_save else args.output

    # Print data summary
    print_data_table(MODELS)

    # Generate plots
    print("\nGenerating plots...")

    if args.plot_type in ['line', 'all']:
        plot_nfe_latency_per_input_size(MODELS, output_path, args.show)

    if args.plot_type in ['combined', 'all']:
        plot_combined_view(MODELS, output_path, args.show)

    if args.plot_type in ['bar', 'all']:
        plot_bar_comparison(MODELS, output_path, args.show)

    print("\nDone!")


if __name__ == '__main__':
    main()
