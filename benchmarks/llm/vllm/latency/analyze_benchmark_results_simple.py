#!/usr/bin/env python3

import csv
import os
from collections import defaultdict

# Read the CSV file
csv_file = 'benchmark_results_summary.csv'
data = []
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Convert string values to appropriate types
        for key in row:
            try:
                row[key] = float(row[key])
            except ValueError:
                pass
        data.append(row)

# Print basic statistics
print("=== Benchmark Results Analysis ===\n")

# Overall statistics
print(f"Total configurations analyzed: {len(data)}")
print(f"Missing configurations: 4 (all TP=8, DP=1 configurations)")

# Best throughput
best_throughput = max(data, key=lambda x: x['overall_throughput_tokens_per_sec'])
print("\n=== Best Throughput Configuration ===")
print(f"TP={int(best_throughput['TP'])}, DP={int(best_throughput['DP'])}, max_tokens={int(best_throughput['max_num_batched_tokens'])}")
print(f"Throughput: {best_throughput['overall_throughput_tokens_per_sec']:.2f} tokens/sec")
print(f"TTFT p50: {best_throughput['ttft_p50_ms']:.2f} ms")
print(f"Inter-token p50: {best_throughput['inter_token_p50_ms']:.2f} ms")

# Best TTFT (time to first token)
best_ttft = min(data, key=lambda x: x['ttft_p50_ms'])
print("\n=== Best Time to First Token (TTFT) Configuration ===")
print(f"TP={int(best_ttft['TP'])}, DP={int(best_ttft['DP'])}, max_tokens={int(best_ttft['max_num_batched_tokens'])}")
print(f"TTFT p50: {best_ttft['ttft_p50_ms']:.2f} ms")
print(f"Throughput: {best_ttft['overall_throughput_tokens_per_sec']:.2f} tokens/sec")
print(f"Inter-token p50: {best_ttft['inter_token_p50_ms']:.2f} ms")

# Best inter-token latency
best_inter = min(data, key=lambda x: x['inter_token_p50_ms'])
print("\n=== Best Inter-Token Latency Configuration ===")
print(f"TP={int(best_inter['TP'])}, DP={int(best_inter['DP'])}, max_tokens={int(best_inter['max_num_batched_tokens'])}")
print(f"Inter-token p50: {best_inter['inter_token_p50_ms']:.2f} ms")
print(f"Throughput: {best_inter['overall_throughput_tokens_per_sec']:.2f} tokens/sec")
print(f"TTFT p50: {best_inter['ttft_p50_ms']:.2f} ms")

# Group by TP and DP
tp_dp_groups = defaultdict(list)
for row in data:
    tp_dp_groups[(int(row['TP']), int(row['DP']))].append(row)

# Find the best max_tokens for each TP/DP combination
print("\n=== Best max_tokens for each TP/DP combination ===")
for (tp, dp), group in sorted(tp_dp_groups.items()):
    best_config = max(group, key=lambda x: x['overall_throughput_tokens_per_sec'])
    print(f"TP={tp}, DP={dp}: max_tokens={int(best_config['max_num_batched_tokens'])}, throughput={best_config['overall_throughput_tokens_per_sec']:.2f} tokens/sec")

# Effect of max_tokens on throughput for different TP/DP combinations
print("\n=== Effect of max_tokens on throughput ===")
for (tp, dp), group in sorted(tp_dp_groups.items()):
    min_tokens = min(group, key=lambda x: x['max_num_batched_tokens'])
    max_tokens = max(group, key=lambda x: x['max_num_batched_tokens'])
    throughput_change = ((max_tokens['overall_throughput_tokens_per_sec'] - min_tokens['overall_throughput_tokens_per_sec']) / 
                         min_tokens['overall_throughput_tokens_per_sec'] * 100)
    print(f"TP={tp}, DP={dp}: {throughput_change:.2f}% change from {int(min_tokens['max_num_batched_tokens'])} to {int(max_tokens['max_num_batched_tokens'])} tokens")

# Create a summary table of the best configurations for different metrics
print("\n=== Summary of Best Configurations ===")
metrics = [
    ('overall_throughput_tokens_per_sec', 'Throughput (tokens/sec)', max),
    ('ttft_p50_ms', 'TTFT p50 (ms)', min),
    ('inter_token_p50_ms', 'Inter-token p50 (ms)', min),
    ('total_latency_p50_ms', 'Total Latency p50 (ms)', min)
]

print(f"{'Metric':<30} {'Best Value':<15} {'TP':<5} {'DP':<5} {'max_tokens':<10}")
print("-" * 70)

for metric, name, func in metrics:
    best_row = func(data, key=lambda x: x[metric])
    best_val = best_row[metric]
    
    print(f"{name:<30} {best_val:>15.2f} {int(best_row['TP']):>5} {int(best_row['DP']):>5} {int(best_row['max_num_batched_tokens']):>10}")

# Save the analysis to a text file
with open('benchmark_analysis.txt', 'w') as f:
    f.write("=== Benchmark Results Analysis ===\n\n")
    f.write(f"Total configurations analyzed: {len(data)}\n")
    f.write(f"Missing configurations: 4 (all TP=8, DP=1 configurations)\n\n")
    
    f.write("=== Best Throughput Configuration ===\n")
    f.write(f"TP={int(best_throughput['TP'])}, DP={int(best_throughput['DP'])}, max_tokens={int(best_throughput['max_num_batched_tokens'])}\n")
    f.write(f"Throughput: {best_throughput['overall_throughput_tokens_per_sec']:.2f} tokens/sec\n")
    f.write(f"TTFT p50: {best_throughput['ttft_p50_ms']:.2f} ms\n")
    f.write(f"Inter-token p50: {best_throughput['inter_token_p50_ms']:.2f} ms\n\n")
    
    f.write("=== Best Time to First Token (TTFT) Configuration ===\n")
    f.write(f"TP={int(best_ttft['TP'])}, DP={int(best_ttft['DP'])}, max_tokens={int(best_ttft['max_num_batched_tokens'])}\n")
    f.write(f"TTFT p50: {best_ttft['ttft_p50_ms']:.2f} ms\n")
    f.write(f"Throughput: {best_ttft['overall_throughput_tokens_per_sec']:.2f} tokens/sec\n")
    f.write(f"Inter-token p50: {best_ttft['inter_token_p50_ms']:.2f} ms\n\n")
    
    f.write("=== Best Inter-Token Latency Configuration ===\n")
    f.write(f"TP={int(best_inter['TP'])}, DP={int(best_inter['DP'])}, max_tokens={int(best_inter['max_num_batched_tokens'])}\n")
    f.write(f"Inter-token p50: {best_inter['inter_token_p50_ms']:.2f} ms\n")
    f.write(f"Throughput: {best_inter['overall_throughput_tokens_per_sec']:.2f} tokens/sec\n")
    f.write(f"TTFT p50: {best_inter['ttft_p50_ms']:.2f} ms\n\n")
    
    f.write("=== Best max_tokens for each TP/DP combination ===\n")
    for (tp, dp), group in sorted(tp_dp_groups.items()):
        best_config = max(group, key=lambda x: x['overall_throughput_tokens_per_sec'])
        f.write(f"TP={tp}, DP={dp}: max_tokens={int(best_config['max_num_batched_tokens'])}, throughput={best_config['overall_throughput_tokens_per_sec']:.2f} tokens/sec\n")
    
    f.write("\n=== Effect of max_tokens on throughput ===\n")
    for (tp, dp), group in sorted(tp_dp_groups.items()):
        min_tokens = min(group, key=lambda x: x['max_num_batched_tokens'])
        max_tokens = max(group, key=lambda x: x['max_num_batched_tokens'])
        throughput_change = ((max_tokens['overall_throughput_tokens_per_sec'] - min_tokens['overall_throughput_tokens_per_sec']) / 
                            min_tokens['overall_throughput_tokens_per_sec'] * 100)
        f.write(f"TP={tp}, DP={dp}: {throughput_change:.2f}% change from {int(min_tokens['max_num_batched_tokens'])} to {int(max_tokens['max_num_batched_tokens'])} tokens\n")
    
    f.write("\n=== Summary of Best Configurations ===\n")
    f.write(f"{'Metric':<30} {'Best Value':<15} {'TP':<5} {'DP':<5} {'max_tokens':<10}\n")
    f.write("-" * 70 + "\n")
    
    for metric, name, func in metrics:
        best_row = func(data, key=lambda x: x[metric])
        best_val = best_row[metric]
        
        f.write(f"{name:<30} {best_val:>15.2f} {int(best_row['TP']):>5} {int(best_row['DP']):>5} {int(best_row['max_num_batched_tokens']):>10}\n")

print(f"\nAnalysis saved to benchmark_analysis.txt")
