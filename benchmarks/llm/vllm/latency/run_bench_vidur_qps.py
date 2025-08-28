#!/usr/bin/env python3
import pandas as pd
import subprocess
import sys
import argparse
import ast

def load_qps_data(csv_path, model_name, device):
    """Load QPS values for given model and device from CSV"""
    df = pd.read_csv(csv_path)
    filtered = df[(df['model_name'] == model_name) & (df['network_device'] == device)]
    
    if filtered.empty:
        print(f"No data found for model {model_name} and device {device}")
        return {}
    
    qps_configs = {}
    for _, row in filtered.iterrows():
        tp = row['tensor_parallel_size']
        replicas = row['num_replicas']
        qps_list = ast.literal_eval(row['qps'])
        qps_configs[(tp, replicas)] = qps_list
    
    return qps_configs

def run_benchmark(model_name, device, tp, dp, qps, max_tokens=3, max_batch_tokens=512, max_num_seqs=512):
    """Run single benchmark configuration"""
    output_dir = f"./vllm_output/vidur_qps/qu_brand/{device}/{model_name.replace('/', '_')}/tp{tp}_dp{dp}/qps{int(qps)}"
    QU_PROMPTS_FILE="/home/ec2-user/s3-local/qu/prompts/validation_results.csv"

    cmd = [
        "python3", "bench_latency.py",
        "--model", model_name,
        "--qps-mode",
        "--qps", str(qps),
        "--tp", str(tp),
        "--dp", str(dp),
        "--max-num-batched-tokens", str(max_batch_tokens),
        "--max-tokens", str(max_tokens),
        "--max-num-seqs", str(max_num_seqs),
        "--qu-prompts-file", f"{QU_PROMPTS_FILE}",
        "--output-dir", output_dir
    ]
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run vLLM benchmarks based on Vidur QPS data")
    parser.add_argument("--model", required=True, help="Model name (e.g., Qwen/Qwen2.5-1.5B)")
    parser.add_argument("--device", required=True, help="Device type (e.g., h100_p5, a10g_g5)")
    parser.add_argument("--csv-path", default="/home/ec2-user/vidur-simulator/analysis/qps_plots/qps_by_model_parallelism_device.csv", help="Path to QPS CSV file")
    parser.add_argument("--total-gpus", type=int, default=8, help="Total GPUs available")
    
    args = parser.parse_args()
    
    # Load QPS configurations
    qps_configs = load_qps_data(args.csv_path, args.model, args.device)
    
    if not qps_configs:
        sys.exit(1)
    
    print(f"Starting benchmark for {args.model} on {args.device}")
    print(f"Found {len(qps_configs)} parallelism configurations")
    print(f"{qps_configs}")
    # Run benchmarks for each configuration
    for (tp, replicas), qps_list in qps_configs.items():
        dp = replicas  # Data parallelism fixed to 1 for single node
        print(f"{(tp,replicas)} have {qps_list} QPS values")
        if tp * dp <= args.total_gpus:
            print(f"\nTesting TP={tp}, Replicas={replicas}")
            
            for qps in qps_list:
                try:
                    run_benchmark(args.model, args.device, tp, dp, qps)
                    print(f"Completed: TP={tp}, QPS={qps}")
                except subprocess.CalledProcessError as e:
                    print(f"Failed: TP={tp}, QPS={qps} - {e}")
        else:
            print(f"Skipping TP={tp} (requires {tp * dp} GPUs, have {args.total_gpus})")
    
    print(f"\nBenchmark sweep completed for {args.model} on {args.device}")

if __name__ == "__main__":
    main()