#!/usr/bin/env python3
"""
Script to plot benchmark comparison between two models.
Takes model names as input for generalizability.
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.gridspec as gridspec

def load_benchmark_data(model_path):
    """Load benchmark data from CSV file."""
    return pd.read_csv(model_path)

def calculate_percentage_diff(value1, value2):
    """Calculate percentage difference between two values."""
    if value1 == 0:
        return float('inf') if value2 != 0 else 0
    return ((value2 - value1) / value1) * 100

def plot_metric_comparison(model1_data, model2_data, model1_name, model2_name, metric, output_dir, higher_is_better=True, tp=1, dp=1, pp=1):
    """
    Plot comparison of a specific metric between two models.
    
    Args:
        model1_data: DataFrame containing model1 benchmark data
        model2_data: DataFrame containing model2 benchmark data
        model1_name: Name of model1 for display
        model2_name: Name of model2 for display
        metric: The metric to plot
        output_dir: Directory to save the plot
        higher_is_better: Whether higher values are better for this metric
        tp: Tensor Parallelism value to filter by
        dp: Data Parallelism value to filter by
        pp: Pipeline Parallelism value to filter by
    """
    # Filter data for specified configuration
    model1_row = model1_data[(model1_data['TP'] == tp) & (model1_data['DP'] == dp) & (model1_data['PP'] == pp)]
    model2_row = model2_data[(model2_data['TP'] == tp) & (model2_data['DP'] == dp) & (model2_data['PP'] == pp)]
    
    if model1_row.empty or model2_row.empty:
        print(f"Warning: No data found for TP={tp}, DP={dp}, PP={pp} configuration for metric {metric}")
        return
    
    model1_value = model1_row[metric].values[0]
    model2_value = model2_row[metric].values[0]
    
    # Calculate percentage difference
    pct_diff = calculate_percentage_diff(model1_value, model2_value)
    
    # Determine if the difference is an improvement or regression
    if higher_is_better:
        is_improvement = pct_diff > 0
    else:
        is_improvement = pct_diff < 0
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars
    bar_width = 0.35
    x = np.array([0, 1])
    bars = ax.bar(x, [model1_value, model2_value], bar_width, 
                 color=['blue', 'green'])
    
    # Add labels and title
    ax.set_xlabel('Models')
    ax.set_ylabel(metric)
    ax.set_title(f'Comparison of {metric} between {model1_name} and {model2_name}')
    ax.set_xticks(x)
    ax.set_xticklabels([model1_name, model2_name])
    
    # Add percentage difference annotation
    color = 'green' if is_improvement else 'red'
    sign = '+' if pct_diff > 0 else ''
    for i, bar in enumerate(bars):
        if i == 1:  # Only annotate the second bar
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (max(model1_value, model2_value) * 0.05),
                    f'{sign}{pct_diff:.2f}%',
                    ha='center', va='bottom', color=color, fontweight='bold')
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{metric}_tp{tp}_dp{dp}_pp{pp}_comparison.png'))
    plt.close()

def plot_all_metrics(model1_data, model2_data, model1_name, model2_name, metrics_config, output_dir, tp=1, dp=1, pp=1):
    """
    Plot all metrics in a single figure for comparison.
    
    Args:
        model1_data: DataFrame containing model1 benchmark data
        model2_data: DataFrame containing model2 benchmark data
        model1_name: Name of model1 for display
        model2_name: Name of model2 for display
        metrics_config: Dictionary of metrics and whether higher is better
        output_dir: Directory to save the plot
        tp: Tensor Parallelism value to filter by
        dp: Data Parallelism value to filter by
        pp: Pipeline Parallelism value to filter by
    """
    # Filter data for specified configuration
    model1_row = model1_data[(model1_data['TP'] == tp) & (model1_data['DP'] == dp) & (model1_data['PP'] == pp)]
    model2_row = model2_data[(model2_data['TP'] == tp) & (model2_data['DP'] == dp) & (model2_data['PP'] == pp)]
    
    if model1_row.empty or model2_row.empty:
        print(f"Warning: No data found for TP={tp}, DP={dp}, PP={pp} configuration")
        return
    
    # Create a figure with subplots for each metric
    n_metrics = len(metrics_config)
    fig = plt.figure(figsize=(15, 4 * n_metrics))
    gs = gridspec.GridSpec(n_metrics, 1)
    
    # Plot each metric
    for i, (metric, higher_is_better) in enumerate(metrics_config.items()):
        ax = fig.add_subplot(gs[i])
        
        model1_value = model1_row[metric].values[0]
        model2_value = model2_row[metric].values[0]
        
        # Calculate percentage difference
        pct_diff = calculate_percentage_diff(model1_value, model2_value)
        
        # Determine if the difference is an improvement or regression
        if higher_is_better:
            is_improvement = pct_diff > 0
        else:
            is_improvement = pct_diff < 0
        
        # Plot bars
        bar_width = 0.35
        x = np.array([0, 1])
        bars = ax.bar(x, [model1_value, model2_value], bar_width, 
                     color=['blue', 'green'])
        
        # Add labels and title
        ax.set_xlabel('Models')
        ax.set_ylabel(metric)
        ax.set_title(f'Comparison of {metric}')
        ax.set_xticks(x)
        ax.set_xticklabels([model1_name, model2_name])
        
        # Add percentage difference annotation
        color = 'green' if is_improvement else 'red'
        sign = '+' if pct_diff > 0 else ''
        for j, bar in enumerate(bars):
            if j == 1:  # Only annotate the second bar
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (max(model1_value, model2_value) * 0.05),
                        f'{sign}{pct_diff:.2f}%',
                        ha='center', va='bottom', color=color, fontweight='bold')
    
    # Add overall title
    plt.suptitle(f'Comparison of all metrics between {model1_name} and {model2_name} (TP={tp}, DP={dp}, PP={pp})', 
                 fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'all_metrics_tp{tp}_dp{dp}_pp{pp}_comparison.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot benchmark comparison between two models')
    parser.add_argument('--model1', required=True, help='First model name (e.g., Qwen3-4B)')
    parser.add_argument('--model2', required=True, help='Second model name (e.g., Qwen3-4B-AWQ)')
    parser.add_argument('--base-dir', default='benchmarks/llm/vllm/latency/sweep_configs', 
                        help='Base directory containing model benchmark results')
    parser.add_argument('--output-dir', default='benchmark_comparison_plots',
                        help='Directory to save the plots')
    parser.add_argument('--tp', type=int, default=1, help='Tensor Parallelism value to filter by')
    parser.add_argument('--dp', type=int, default=1, help='Data Parallelism value to filter by')
    parser.add_argument('--pp', type=int, default=1, help='Pipeline Parallelism value to filter by')
    parser.add_argument('--all-configs', action='store_true', help='Plot all available configurations')
    parser.add_argument('--combined', action='store_true', help='Create a combined plot with all metrics')
    
    args = parser.parse_args()
    
    # Construct paths to benchmark files
    model1_dir = f"Qwen_{args.model1}"
    model2_dir = f"Qwen_{args.model2}"
    
    model1_path = os.path.join(args.base_dir, model1_dir, 'benchmark_results_summary.csv')
    model2_path = os.path.join(args.base_dir, model2_dir, 'benchmark_results_summary.csv')
    
    # Check if files exist
    if not os.path.exists(model1_path):
        print(f"Error: Benchmark file for {args.model1} not found at {model1_path}")
        return
    
    if not os.path.exists(model2_path):
        print(f"Error: Benchmark file for {args.model2} not found at {model2_path}")
        return
    
    # Load benchmark data
    model1_data = load_benchmark_data(model1_path)
    model2_data = load_benchmark_data(model2_path)
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, f"{args.model1}_vs_{args.model2}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define metrics to plot and whether higher values are better
    metrics_config = {
        'overall_throughput_tokens_per_sec': True,  # Higher is better
        'total_latency_p50_ms': False,  # Lower is better
        'total_latency_p99_ms': False,  # Lower is better
        'ttft_p50_ms': False,  # Lower is better
        'ttft_p99_ms': False,  # Lower is better
        'inter_token_p50_ms': False,  # Lower is better
        'inter_token_p99_ms': False  # Lower is better
    }
    
    # Plot each metric
    if args.all_configs:
        # Get all unique configurations
        configs = []
        for df in [model1_data, model2_data]:
            for _, row in df.iterrows():
                config = (row['TP'], row['DP'], row['PP'])
                if config not in configs:
                    configs.append(config)
        
        # Plot each metric for each configuration
        for tp, dp, pp in configs:
            config_output_dir = os.path.join(output_dir, f"tp{tp}_dp{dp}_pp{pp}")
            os.makedirs(config_output_dir, exist_ok=True)
            
            # Individual metric plots
            for metric, higher_is_better in metrics_config.items():
                plot_metric_comparison(
                    model1_data, model2_data, 
                    args.model1, args.model2, 
                    metric, config_output_dir, 
                    higher_is_better, tp, dp, pp
                )
            
            # Combined plot if requested
            if args.combined:
                plot_all_metrics(
                    model1_data, model2_data,
                    args.model1, args.model2,
                    metrics_config, config_output_dir,
                    tp, dp, pp
                )
    else:
        # Plot each metric for the specified configuration
        for metric, higher_is_better in metrics_config.items():
            plot_metric_comparison(
                model1_data, model2_data, 
                args.model1, args.model2, 
                metric, output_dir, 
                higher_is_better, args.tp, args.dp, args.pp
            )
        
        # Combined plot if requested
        if args.combined:
            plot_all_metrics(
                model1_data, model2_data,
                args.model1, args.model2,
                metrics_config, output_dir,
                args.tp, args.dp, args.pp
            )
    
    print(f"Plots have been saved to {output_dir}")

if __name__ == "__main__":
    main()
