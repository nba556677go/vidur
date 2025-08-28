#!/bin/bash

# ==============================================================================
# vLLM Benchmark Automation Script
#
# This script systematically runs the bench_latency.py script against a series
# of configurations to find the optimal performance settings for a given model
# on an 8-GPU system.
#
# It iterates through:
#   - All valid Tensor Parallelism (tp) and Data Parallelism (dp) combinations.
#   - A range of `max-num-batched-tokens` values.
#
# ==============================================================================

# --- Script Configuration ---

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# Pipes fail on the first command that fails, not the last.
set -o pipefail

# --- Benchmark Parameters ---

# Fixed parameters for every benchmark run
MODEL_NAME="meta-llama/Meta-Llama-3-8B"
#PROMPTS_FILE="../../prompts/prompt_extend_4000_numprompts100.txt"
CONCURRENCY=1
TOTAL_GPUS=8
OUTPUT_DIR="./sweep_configs"
TRACE="./vidur_output/request_metrics.csv"
PROMPT_TEMPLATE="../../prompts/prompt_numprompts150.txt" #still have bug. skip now
#
# Base command for the Python benchmark script
#BASE_CMD="python3 bench_latency.py --model $MODEL_NAME --trace $TRACE --concurrency $CONCURRENCY --output-dir $OUTPUT_DIR --prompt-templates-file $PROMPT_TEMPLATE"
BASE_CMD="python3 bench_latency.py --model $MODEL_NAME --trace $TRACE --concurrency $CONCURRENCY --output-dir $OUTPUT_DIR"
# --- Iteration Values ---

# Valid Tensor Parallelism (tp) sizes to test
#TP_VALUES=(1 2 4 8)
TP_VALUES=(1)
# Valid Data Parallelism (dp) sizes to test
#DP_VALUES=(1 2 4 8)
DP_VALUES=(1 8)

# Define a specific set of 4 values for max-num-batched-tokens to test.
# This samples performance across the requested range of 4096 to 75360.
#TOKEN_BATCH_VALUES=(4096 20480 40960 73728)
TOKEN_BATCH_VALUES=(4096)


# --- Main Loop ---

echo "Starting vLLM benchmark sweep..."
echo "Total GPUs available: $TOTAL_GPUS"

# Loop over Tensor Parallelism values
for tp in "${TP_VALUES[@]}"; do
  # Loop over Data Parallelism values
  for dp in "${DP_VALUES[@]}"; do
    # Check if the TP * DP combination is valid for the available hardware
    if (( tp * dp <= TOTAL_GPUS )); then
      
      echo ""
      echo "------------------------------------------------------------------"
      echo "Testing Parallelism Config: TP=$tp, DP=$dp"
      echo "------------------------------------------------------------------"

      # Loop over the specified max-num-batched-tokens values
      for max_tokens in "${TOKEN_BATCH_VALUES[@]}"; do
        
        echo ""
        echo "===== RUNNING: TP=$tp, DP=$dp, max_tokens=$max_tokens ====="

        # Construct the full command
        full_command="$BASE_CMD --tp $tp --dp $dp --max-num-batched-tokens $max_tokens"

        # Print the command being executed
        echo "Executing: $full_command"
        
        # Execute the benchmark command
        $full_command
        
        echo "===== COMPLETED: TP=$tp, DP=$dp, max_tokens=$max_tokens ====="
        # Optional: Add a small delay between runs if needed
        # sleep 5
      done

    else
      echo ""
      echo "--- SKIPPING: TP=$tp, DP=$dp (requires $((tp * dp)) GPUs, have $TOTAL_GPUS) ---"
    fi
  done
done

echo ""
echo "=================================================================="
echo "Benchmark sweep finished!"
echo "=================================================================="