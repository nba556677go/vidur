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
NUM_PROMPTS=2000
#output token size
MAX_TOKENS=3 
# Fixed parameters for every benchmark run
#MODEL_NAME="meta-llama/Meta-Llama-3-8B"
MODEL_NAME="Qwen/Qwen2.5-1.5B"
#PROMPTS_FILE="../../prompts/prompt_extend_2048_numprompts150.txt"
PROMPTS_FILE="../../prompts/prompt_extend_152_numprompts2000.txt"

#qu promptfile
QU_PROMPTS_FILE="/home/ec2-user/s3-local/qu/prompts/validation_results.csv"
CONCURRENCY=30
TOTAL_GPUS=8
#OUTPUT_DIR="./sweep_configs/a100_40g"
OUTPUT_ROOT_DIR="qu_brand"
# Base command for the Python benchmark script
#QPS=0 - use prompy mode
#QPS_VALUES=(0.25 0.5 2 8)
QPS_VALUES=(29)

for qps in "${QPS_VALUES[@]}"; do
    #OUTPUT_DIR add if QU_PROMPTS_FILE is empty use the current OUTPUT_DIR, else replace numprompts with qu_brand string
    if $QU_PROMPTS_FILE == ""; then
        OUTPUT_DIR="./vllm_output/$OUTPUT_ROOT_DIR/numprompts$NUM_PROMPTS/qps$qps"
    else
        OUTPUT_DIR="./vllm_output/$OUTPUT_ROOT_DIR/qu_brand/qps$qps"
    fi
    if [ -n "${qps:-}" ]; then
        BASE_CMD="python3 bench_latency.py \
          --model $MODEL_NAME \
          --qps-mode \
          --qu-prompts-file $QU_PROMPTS_FILE \
          --qps $qps \
          --output-dir $OUTPUT_DIR"
    else
        BASE_CMD="python3 bench_latency.py \
          --model $MODEL_NAME \
          --prompts-file $PROMPTS_FILE \
          --concurrency $CONCURRENCY \
          --output-dir $OUTPUT_DIR"
    fi

    # --- Iteration Values ---

    # Valid Tensor Parallelism (tp) sizes to test
    #TP_VALUES=(1 2 4 8)
    TP_VALUES=(2)
    # Valid Data Parallelism (dp) sizes to test
    #DP_VALUES=(1 2 4 8)
    DP_VALUES=(1)

    # Define a specific set of 4 values for max-num-batched-tokens to test.
    # This samples performance across the requested range of 4096 to 75360.
    #TOKEN_BATCH_VALUES=(4096 20480 40960 73728)
    TOKEN_BATCH_VALUES=(512)

    #fix engine configs 
    MAX_NUM_SEQS=512 # concurrent
    
    # --- Main Loop ---

    echo "Starting vLLM benchmark sweep for QPS=$qps..."
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
          for max_batch_tokens in "${TOKEN_BATCH_VALUES[@]}"; do
            
            echo ""
            echo "===== RUNNING: TP=$tp, DP=$dp, max_batch_tokens=$max_batch_tokens, max_tokens=$MAX_TOKENS ====="

            # Construct the full command
            full_command="$BASE_CMD --tp $tp --dp $dp --max-num-batched-tokens $max_batch_tokens --max-tokens $MAX_TOKENS --max-num-seqs $MAX_NUM_SEQS"

            # Print the command being executed
            echo "Executing: $full_command"
            
            # Execute the benchmark command
            $full_command
            
            echo "===== COMPLETED: TP=$tp, DP=$dp, max_batch_tokens=$max_batch_tokens ====="
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
    echo "Benchmark sweep finished for QPS=$qps!"
    echo "=================================================================="

done
