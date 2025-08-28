#!/usr/bin/env python3

import os
import json
import glob
import csv
from pathlib import Path

# Path to the sweep_configs directory
sweep_configs_dir = "/Users/binghann/Documents/s3-local/benchmarks/llm/vllm/latency/sweep_configs/Qwen_Qwen3-4B-AWQ"

# Find all benchmark_results.json files
result_files = glob.glob(f"{sweep_configs_dir}/run_*/benchmark_results.json")

# CSV header
csv_header = [
    "TP", "DP", "PP", "max_num_batched_tokens", 
    "overall_throughput_tokens_per_sec",
    "total_latency_p50_ms", "total_latency_p99_ms",
    "ttft_p50_ms", "ttft_p99_ms",
    "inter_token_p50_ms", "inter_token_p99_ms"
]

# List to store all results
all_results = []

# Process each result file
for result_file in result_files:
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
            
        # Extract configuration
        config = data["benchmark_config"]
        tp = config["tensor_parallel_size"]
        dp = config["data_parallel_size"]
        pp = config["pipeline_parallel_size"]
        max_tokens = config["max_num_batched_tokens"]
        
        # Extract metrics
        stats = data["benchmark_stats"]
        throughput = stats["summary"]["overall_throughput_tokens_per_sec"]
        
        latency_percentiles = stats["latency_percentiles_ms"]
        total_latency_p50 = latency_percentiles["total_generation_latency"]["total_p50"]
        total_latency_p99 = latency_percentiles["total_generation_latency"]["total_p99"]
        
        ttft_p50 = latency_percentiles["time_to_first_token"]["first_token_p50"]
        ttft_p99 = latency_percentiles["time_to_first_token"]["first_token_p99"]
        
        inter_token_p50 = latency_percentiles["inter_token_latency"]["inter_token_p50"]
        inter_token_p99 = latency_percentiles["inter_token_latency"]["inter_token_p99"]
        
        # Add to results list
        all_results.append([
            tp, dp, pp, max_tokens,
            throughput,
            total_latency_p50, total_latency_p99,
            ttft_p50, ttft_p99,
            inter_token_p50, inter_token_p99
        ])
        
        print(f"Processed: TP={tp}, DP={dp}, max_tokens={max_tokens}")
        
    except Exception as e:
        run_dir = os.path.basename(os.path.dirname(result_file))
        print(f"Error processing {run_dir}: {e}")

# Sort results by TP, DP, and max_tokens
all_results.sort(key=lambda x: (x[0], x[1], x[3]))

# Write to CSV
csv_file = os.path.join(sweep_configs_dir, "benchmark_results_summary.csv")
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_header)
    writer.writerows(all_results)

print(f"\nSummary written to {csv_file}")

# Print missing configurations
print("\nMissing configurations:")
expected_configs = []
for tp in [1, 2, 4, 8]:
    for dp in [1, 2, 4, 8]:
        if tp * dp <= 8:
            for max_tokens in [4096, 20480, 40960, 73728]:
                expected_configs.append((tp, dp, max_tokens))

existing_configs = [(r[0], r[1], r[3]) for r in all_results]
for config in expected_configs:
    if config not in existing_configs:
        print(f"  TP={config[0]}, DP={config[1]}, max_tokens={config[2]}")
