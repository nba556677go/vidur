import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Path to the sweep_configs directory
sweep_configs_dir = "/Users/binghann/Documents/s3-local/benchmarks/llm/vllm/latency/sweep_configs/Qwen_Qwen3-4B-AWQ"

# Read the CSV file
df = pd.read_csv(os.path.join(sweep_configs_dir, 'benchmark_results_summary.csv'))

# Define the metrics to analyze
metrics = [
    'overall_throughput_tokens_per_sec',
    'total_latency_p50_ms',
    'total_latency_p99_ms',
    'ttft_p50_ms',
    'ttft_p99_ms',
    'inter_token_p50_ms',
    'inter_token_p99_ms'
]

# Create output directory if it doesn't exist
plots_dir = os.path.join(sweep_configs_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# For each metric, find the best and worst configurations
for metric in metrics:
    # For throughput, higher is better; for latency, lower is better
    is_throughput = 'throughput' in metric
    
    if is_throughput:
        best_config = df.loc[df[metric].idxmax()]
        worst_config = df.loc[df[metric].idxmin()]
    else:
        best_config = df.loc[df[metric].idxmin()]
        worst_config = df.loc[df[metric].idxmax()]
    
    # Calculate percentage difference
    if is_throughput:
        pct_diff = (best_config[metric] - worst_config[metric]) / worst_config[metric] * 100
    else:
        pct_diff = (worst_config[metric] - best_config[metric]) / worst_config[metric] * 100
    
    # Create a figure
    plt.figure(figsize=(12, 8))
    
    # Create labels for x-axis
    best_label = f"TP={best_config['TP']}, DP={best_config['DP']}, Max Tokens={best_config['max_num_batched_tokens']}"
    worst_label = f"TP={worst_config['TP']}, DP={worst_config['DP']}, Max Tokens={worst_config['max_num_batched_tokens']}"
    
    # Create bar plot
    x = np.arange(2)
    bars = plt.bar(x, [best_config[metric], worst_config[metric]], width=0.6)
    
    # Add percentage difference annotation
    plt.annotate(f'+{pct_diff:.2f}%' if is_throughput else f'-{pct_diff:.2f}%',
                xy=(0.5, max(best_config[metric], worst_config[metric]) * 1.05),
                ha='center', va='bottom',
                fontsize=14, fontweight='bold',
                color='green' if pct_diff > 0 else 'red')
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height * 1.01,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=12)
    
    # Set labels and title
    plt.xticks(x, [best_label, worst_label], rotation=15, ha='right')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f'Best vs Worst Configuration for {metric.replace("_", " ").title()}')
    
    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(plots_dir, f'{metric}_comparison.png'), dpi=300)
    plt.close()

# Create a combined plot showing all metrics normalized
plt.figure(figsize=(15, 10))

# For each metric, calculate the normalized values
bar_positions = np.arange(len(metrics))
bar_width = 0.35
best_values = []
worst_values = []
pct_diffs = []

for i, metric in enumerate(metrics):
    is_throughput = 'throughput' in metric
    
    if is_throughput:
        best_config = df.loc[df[metric].idxmax()]
        worst_config = df.loc[df[metric].idxmin()]
        # For throughput, normalize so that best is 1.0
        best_norm = 1.0
        worst_norm = worst_config[metric] / best_config[metric]
        pct_diff = (best_config[metric] - worst_config[metric]) / worst_config[metric] * 100
    else:
        best_config = df.loc[df[metric].idxmin()]
        worst_config = df.loc[df[metric].idxmax()]
        # For latency, normalize so that best is 1.0
        best_norm = 1.0
        worst_norm = worst_config[metric] / best_config[metric]
        pct_diff = (worst_config[metric] - best_config[metric]) / worst_config[metric] * 100
    
    best_values.append(best_norm)
    worst_values.append(worst_norm)
    pct_diffs.append(pct_diff)

# Create bar plot for normalized values
plt.bar(bar_positions - bar_width/2, best_values, bar_width, label='Best Config', color='green')
plt.bar(bar_positions + bar_width/2, worst_values, bar_width, label='Worst Config', color='red')

# Add percentage difference annotations
for i, (pos, pct) in enumerate(zip(bar_positions, pct_diffs)):
    plt.annotate(f'{pct:.2f}%',
                xy=(pos, max(best_values[i], worst_values[i]) + 0.05),
                ha='center', va='bottom',
                fontsize=12, fontweight='bold',
                color='green')

# Set labels and title
plt.xlabel('Metrics')
plt.ylabel('Normalized Value (Best = 1.0)')
plt.title('Normalized Comparison of Best vs Worst Configurations Across All Metrics')
plt.xticks(bar_positions, [m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(plots_dir, 'all_metrics_normalized.png'), dpi=300)

# Create a summary table with configurations
summary_data = []
for metric in metrics:
    is_throughput = 'throughput' in metric
    
    if is_throughput:
        best_config = df.loc[df[metric].idxmax()]
        worst_config = df.loc[df[metric].idxmin()]
        pct_diff = (best_config[metric] - worst_config[metric]) / worst_config[metric] * 100
    else:
        best_config = df.loc[df[metric].idxmin()]
        worst_config = df.loc[df[metric].idxmax()]
        pct_diff = (worst_config[metric] - best_config[metric]) / worst_config[metric] * 100
    
    summary_data.append({
        'Metric': metric,
        'Best_TP': best_config['TP'],
        'Best_DP': best_config['DP'],
        'Best_MaxTokens': best_config['max_num_batched_tokens'],
        'Best_Value': best_config[metric],
        'Worst_TP': worst_config['TP'],
        'Worst_DP': worst_config['DP'],
        'Worst_MaxTokens': worst_config['max_num_batched_tokens'],
        'Worst_Value': worst_config[metric],
        'Pct_Improvement': pct_diff
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(plots_dir, 'performance_summary.csv'), index=False)

print(f"All plots and summary have been generated in the '{plots_dir}' directory.")
