#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Read the CSV file
df = pd.read_csv('benchmark_results_summary.csv')

# Print basic statistics
print("=== Benchmark Results Analysis ===\n")

# Overall statistics
print(f"Total configurations analyzed: {len(df)}")
print(f"Missing configurations: 4 (all TP=8, DP=1 configurations)")

# Best throughput
best_throughput = df.loc[df['overall_throughput_tokens_per_sec'].idxmax()]
print("\n=== Best Throughput Configuration ===")
print(f"TP={best_throughput['TP']}, DP={best_throughput['DP']}, max_tokens={best_throughput['max_num_batched_tokens']}")
print(f"Throughput: {best_throughput['overall_throughput_tokens_per_sec']:.2f} tokens/sec")
print(f"TTFT p50: {best_throughput['ttft_p50_ms']:.2f} ms")
print(f"Inter-token p50: {best_throughput['inter_token_p50_ms']:.2f} ms")

# Best TTFT (time to first token)
best_ttft = df.loc[df['ttft_p50_ms'].idxmin()]
print("\n=== Best Time to First Token (TTFT) Configuration ===")
print(f"TP={best_ttft['TP']}, DP={best_ttft['DP']}, max_tokens={best_ttft['max_num_batched_tokens']}")
print(f"TTFT p50: {best_ttft['ttft_p50_ms']:.2f} ms")
print(f"Throughput: {best_ttft['overall_throughput_tokens_per_sec']:.2f} tokens/sec")
print(f"Inter-token p50: {best_ttft['inter_token_p50_ms']:.2f} ms")

# Best inter-token latency
best_inter = df.loc[df['inter_token_p50_ms'].idxmin()]
print("\n=== Best Inter-Token Latency Configuration ===")
print(f"TP={best_inter['TP']}, DP={best_inter['DP']}, max_tokens={best_inter['max_num_batched_tokens']}")
print(f"Inter-token p50: {best_inter['inter_token_p50_ms']:.2f} ms")
print(f"Throughput: {best_inter['overall_throughput_tokens_per_sec']:.2f} tokens/sec")
print(f"TTFT p50: {best_inter['ttft_p50_ms']:.2f} ms")

# Group by TP and DP, and find the best max_tokens for each combination
print("\n=== Best max_tokens for each TP/DP combination ===")
for (tp, dp), group in df.groupby(['TP', 'DP']):
    best_config = group.loc[group['overall_throughput_tokens_per_sec'].idxmax()]
    print(f"TP={tp}, DP={dp}: max_tokens={best_config['max_num_batched_tokens']}, throughput={best_config['overall_throughput_tokens_per_sec']:.2f} tokens/sec")

# Effect of max_tokens on throughput for different TP/DP combinations
print("\n=== Effect of max_tokens on throughput ===")
for (tp, dp), group in df.groupby(['TP', 'DP']):
    min_tokens = group.loc[group['max_num_batched_tokens'].idxmin()]
    max_tokens = group.loc[group['max_num_batched_tokens'].idxmax()]
    throughput_change = ((max_tokens['overall_throughput_tokens_per_sec'] - min_tokens['overall_throughput_tokens_per_sec']) / 
                         min_tokens['overall_throughput_tokens_per_sec'] * 100)
    print(f"TP={tp}, DP={dp}: {throughput_change:.2f}% change from {min_tokens['max_num_batched_tokens']} to {max_tokens['max_num_batched_tokens']} tokens")

# Create a summary table of the best configurations for different metrics
print("\n=== Summary of Best Configurations ===")
metrics = ['overall_throughput_tokens_per_sec', 'ttft_p50_ms', 'inter_token_p50_ms', 'total_latency_p50_ms']
metric_names = ['Throughput (tokens/sec)', 'TTFT p50 (ms)', 'Inter-token p50 (ms)', 'Total Latency p50 (ms)']
best_configs = []

for i, metric in enumerate(metrics):
    if 'latency' in metric or 'ttft' in metric or 'inter_token' in metric:
        best_row = df.loc[df[metric].idxmin()]
        best_val = best_row[metric]
    else:
        best_row = df.loc[df[metric].idxmax()]
        best_val = best_row[metric]
    
    best_configs.append({
        'Metric': metric_names[i],
        'Best Value': f"{best_val:.2f}",
        'TP': best_row['TP'],
        'DP': best_row['DP'],
        'max_tokens': best_row['max_num_batched_tokens']
    })

summary_df = pd.DataFrame(best_configs)
print(summary_df.to_string(index=False))

# Save the analysis to a text file
with open('benchmark_analysis.txt', 'w') as f:
    f.write("=== Benchmark Results Analysis ===\n\n")
    f.write(f"Total configurations analyzed: {len(df)}\n")
    f.write(f"Missing configurations: 4 (all TP=8, DP=1 configurations)\n\n")
    
    f.write("=== Best Throughput Configuration ===\n")
    f.write(f"TP={best_throughput['TP']}, DP={best_throughput['DP']}, max_tokens={best_throughput['max_num_batched_tokens']}\n")
    f.write(f"Throughput: {best_throughput['overall_throughput_tokens_per_sec']:.2f} tokens/sec\n")
    f.write(f"TTFT p50: {best_throughput['ttft_p50_ms']:.2f} ms\n")
    f.write(f"Inter-token p50: {best_throughput['inter_token_p50_ms']:.2f} ms\n\n")
    
    f.write("=== Best Time to First Token (TTFT) Configuration ===\n")
    f.write(f"TP={best_ttft['TP']}, DP={best_ttft['DP']}, max_tokens={best_ttft['max_num_batched_tokens']}\n")
    f.write(f"TTFT p50: {best_ttft['ttft_p50_ms']:.2f} ms\n")
    f.write(f"Throughput: {best_ttft['overall_throughput_tokens_per_sec']:.2f} tokens/sec\n")
    f.write(f"Inter-token p50: {best_ttft['inter_token_p50_ms']:.2f} ms\n\n")
    
    f.write("=== Best Inter-Token Latency Configuration ===\n")
    f.write(f"TP={best_inter['TP']}, DP={best_inter['DP']}, max_tokens={best_inter['max_num_batched_tokens']}\n")
    f.write(f"Inter-token p50: {best_inter['inter_token_p50_ms']:.2f} ms\n")
    f.write(f"Throughput: {best_inter['overall_throughput_tokens_per_sec']:.2f} tokens/sec\n")
    f.write(f"TTFT p50: {best_inter['ttft_p50_ms']:.2f} ms\n\n")
    
    f.write("=== Best max_tokens for each TP/DP combination ===\n")
    for (tp, dp), group in df.groupby(['TP', 'DP']):
        best_config = group.loc[group['overall_throughput_tokens_per_sec'].idxmax()]
        f.write(f"TP={tp}, DP={dp}: max_tokens={best_config['max_num_batched_tokens']}, throughput={best_config['overall_throughput_tokens_per_sec']:.2f} tokens/sec\n")
    
    f.write("\n=== Effect of max_tokens on throughput ===\n")
    for (tp, dp), group in df.groupby(['TP', 'DP']):
        min_tokens = group.loc[group['max_num_batched_tokens'].idxmin()]
        max_tokens = group.loc[group['max_num_batched_tokens'].idxmax()]
        throughput_change = ((max_tokens['overall_throughput_tokens_per_sec'] - min_tokens['overall_throughput_tokens_per_sec']) / 
                            min_tokens['overall_throughput_tokens_per_sec'] * 100)
        f.write(f"TP={tp}, DP={dp}: {throughput_change:.2f}% change from {min_tokens['max_num_batched_tokens']} to {max_tokens['max_num_batched_tokens']} tokens\n")
    
    f.write("\n=== Summary of Best Configurations ===\n")
    f.write(summary_df.to_string(index=False))

print(f"\nAnalysis saved to benchmark_analysis.txt")

# Try to create some visualizations if matplotlib is available
try:
    # Create a directory for plots
    plots_dir = Path("benchmark_plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Plot throughput by TP/DP combination
    plt.figure(figsize=(12, 8))
    for (tp, dp), group in df.groupby(['TP', 'DP']):
        plt.plot(group['max_num_batched_tokens'], group['overall_throughput_tokens_per_sec'], 
                marker='o', label=f'TP={tp}, DP={dp}')
    
    plt.xlabel('Max Num Batched Tokens')
    plt.ylabel('Throughput (tokens/sec)')
    plt.title('Throughput vs Max Num Batched Tokens')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'throughput_vs_tokens.png')
    
    # Plot TTFT by TP/DP combination
    plt.figure(figsize=(12, 8))
    for (tp, dp), group in df.groupby(['TP', 'DP']):
        plt.plot(group['max_num_batched_tokens'], group['ttft_p50_ms'], 
                marker='o', label=f'TP={tp}, DP={dp}')
    
    plt.xlabel('Max Num Batched Tokens')
    plt.ylabel('Time to First Token p50 (ms)')
    plt.title('TTFT vs Max Num Batched Tokens')
    plt.xscale('log')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'ttft_vs_tokens.png')
    
    print(f"Plots saved to {plots_dir}/")
except Exception as e:
    print(f"Could not create plots: {e}")
