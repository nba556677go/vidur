#!/usr/bin/env python3

import csv
import os

# Path to the sweep_configs directory
sweep_configs_dir = "/Users/binghann/Documents/s3-local/benchmarks/llm/vllm/latency/sweep_configs/Qwen_Qwen3-4B-AWQ"

# Read the benchmark results CSV
input_csv = os.path.join(sweep_configs_dir, 'benchmark_results_summary.csv')
output_csv = os.path.join(sweep_configs_dir, 'benchmark_metrics_comparison.csv')

# Define the metrics we want to include in the output
output_metrics = [
    'TP', 'DP', 'max_num_batched_tokens', 
    'overall_throughput_tokens_per_sec',
    'ttft_p50_ms', 'inter_token_p50_ms',
    'total_latency_p50_ms'
]

# Read the input CSV
data = []
with open(input_csv, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Convert string values to appropriate types
        for key in row:
            try:
                if key in ['TP', 'DP', 'PP', 'max_num_batched_tokens']:
                    row[key] = int(float(row[key]))
                else:
                    row[key] = float(row[key])
            except ValueError:
                pass
        data.append(row)

# Sort the data by TP, DP, and max_tokens
data.sort(key=lambda x: (x['TP'], x['DP'], x['max_num_batched_tokens']))

# Write the output CSV
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    
    # Write header
    header = [
        'TP', 'DP', 'max_tokens', 
        'throughput (tokens/sec)',
        'TTFT p50 (ms)', 'inter-token p50 (ms)',
        'total latency p50 (ms)'
    ]
    writer.writerow(header)
    
    # Write data
    for row in data:
        output_row = [
            row['TP'],
            row['DP'],
            row['max_num_batched_tokens'],
            round(row['overall_throughput_tokens_per_sec'], 2),
            round(row['ttft_p50_ms'], 2),
            round(row['inter_token_p50_ms'], 2),
            round(row['total_latency_p50_ms'], 2)
        ]
        writer.writerow(output_row)

print(f"Metrics comparison CSV created: {output_csv}")

# Also create a text file with the top configurations for each metric
top_config_file = os.path.join(sweep_configs_dir, 'top_configurations.txt')
with open(top_config_file, 'w') as f:
    # Top throughput configurations
    f.write("=== Top 5 Configurations by Throughput ===\n")
    top_throughput = sorted(data, key=lambda x: x['overall_throughput_tokens_per_sec'], reverse=True)[:5]
    for i, row in enumerate(top_throughput, 1):
        f.write(f"{i}. TP={row['TP']}, DP={row['DP']}, max_tokens={row['max_num_batched_tokens']}: {row['overall_throughput_tokens_per_sec']:.2f} tokens/sec\n")
    
    # Top TTFT configurations
    f.write("\n=== Top 5 Configurations by Time to First Token (lowest) ===\n")
    top_ttft = sorted(data, key=lambda x: x['ttft_p50_ms'])[:5]
    for i, row in enumerate(top_ttft, 1):
        f.write(f"{i}. TP={row['TP']}, DP={row['DP']}, max_tokens={row['max_num_batched_tokens']}: {row['ttft_p50_ms']:.2f} ms\n")
    
    # Top inter-token latency configurations
    f.write("\n=== Top 5 Configurations by Inter-Token Latency (lowest) ===\n")
    top_inter = sorted(data, key=lambda x: x['inter_token_p50_ms'])[:5]
    for i, row in enumerate(top_inter, 1):
        f.write(f"{i}. TP={row['TP']}, DP={row['DP']}, max_tokens={row['max_num_batched_tokens']}: {row['inter_token_p50_ms']:.2f} ms\n")
    
    # Top total latency configurations
    f.write("\n=== Top 5 Configurations by Total Latency (lowest) ===\n")
    top_latency = sorted(data, key=lambda x: x['total_latency_p50_ms'])[:5]
    for i, row in enumerate(top_latency, 1):
        f.write(f"{i}. TP={row['TP']}, DP={row['DP']}, max_tokens={row['max_num_batched_tokens']}: {row['total_latency_p50_ms']:.2f} ms\n")

print(f"Top configurations summary created: {top_config_file}")
