#!/usr/bin/env python3
from vllmbenchparser_fixedqps import VLLMBenchParser
import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict

class VLLMBenchQu(VLLMBenchParser):
    def __init__(self, base_dir: str, output_dir: str):
        super().__init__(base_dir, output_dir)
    
    def calculate_process_time_from_log(self, benchmark_log_path: str) -> Dict[str, float]:
        """Calculate process times from benchmark.log by finding timestamps"""
        try:
            with open(benchmark_log_path, 'r') as f:
                lines = f.readlines()
            
            init_start_time = None
            benchmark_start_time = None
            end_time = None
            
            for line in lines:
                if "Initializing vLLM engine" in line:
                    timestamp_str = line.split(' [')[0]
                    init_start_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
                elif "Starting QPS mode benchmark" in line:
                    timestamp_str = line.split(' [')[0]
                    benchmark_start_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
                elif "Benchmark results saved" in line:
                    timestamp_str = line.split(' [')[0]
                    end_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
            
            result = {'process_time': 0.0, 'total_time_with_init': 0.0}
            
            if benchmark_start_time and end_time:
                result['process_time'] = (end_time - benchmark_start_time).total_seconds()
            
            if init_start_time and end_time:
                result['total_time_with_init'] = (end_time - init_start_time).total_seconds()
            
            return result
            
        except Exception as e:
            print(f"Error processing {benchmark_log_path}: {e}")
            return {'process_time': 0.0, 'total_time_with_init': 0.0}
    
    def parse_all(self) -> pd.DataFrame:
        """Parse all directories recursively to find benchmark results"""
        results_list = []
        
        # Walk through all directories recursively
        for root, dirs, files in os.walk(self.base_dir):
            # Check if this directory contains benchmark results
            if 'config.json' in files and 'benchmark_results.json' in files:
                metrics, config = self.process_directory(root)
                
                if metrics and config:
                    # Calculate process time from benchmark.log if available
                    benchmark_log_path = os.path.join(root, 'benchmark.log')
                    timing_data = {'process_time': 0.0, 'total_time_with_init': 0.0}
                    if os.path.exists(benchmark_log_path):
                        timing_data = self.calculate_process_time_from_log(benchmark_log_path)
                    
                    # Parse path to extract device, qps, and model info
                    # Path format: base_dir/device/model/tp_dp/qps/model/run_timestamp
                    path_parts = root.replace(self.base_dir, '').strip('/').split('/')
                    
                    device = None
                    qps_value = None
                    model_from_path = None
                    
                    # Extract info from path parts based on position
                    if len(path_parts) >= 1:
                        device = path_parts[0]  # First part is device
                    if len(path_parts) >= 2:
                        model_from_path = path_parts[1].replace('_', '/')  # Second part is model
                    if len(path_parts) >= 4 and path_parts[3].startswith('qps'):
                        qps_value = float(path_parts[3].replace('qps', ''))  # Fourth part is qps
                    
                    results_list.append({
                        'Device': device,
                        'QPS_Dir': qps_value,
                        'Model_Path': model_from_path,
                        'Run_Path': root,
                        'Model': config.model_name,
                        'Num_Replicas': config.num_replicas,
                        'Tensor_Parallel': config.tensor_parallel_size,
                        'Pipeline_Parallel': config.num_pipeline_stages,
                        'Max_Batch_Tokens': config.max_num_batched_tokens,
                        'Max_Num_Seqs': config.max_num_seqs,
                        'Concurrency': config.concurrency,
                        'QPS': config.qps,
                        'Max_Tokens': config.max_tokens,
                        'Temperature': config.temperature,
                        'Top_P': config.top_p,
                        'Total_Gen_Avg': metrics['total_gen_avg'],
                        'Total_Gen_P50': metrics['total_gen_p50'],
                        'Total_Gen_P90': metrics['total_gen_p90'],
                        'Total_Gen_P99': metrics['total_gen_p99'],
                        'First_Token_Avg': metrics['first_token_avg'],
                        'First_Token_P50': metrics['first_token_p50'],
                        'First_Token_P90': metrics['first_token_p90'],
                        'First_Token_P99': metrics['first_token_p99'],
                        'Inter_Token_Avg': metrics['inter_token_avg'],
                        'Inter_Token_P50': metrics['inter_token_p50'],
                        'Inter_Token_P90': metrics['inter_token_p90'],
                        'Inter_Token_P99': metrics['inter_token_p99'],
                        'Schedule_Delay_Avg': metrics['schedule_delay_avg'],
                        'Schedule_Delay_P50': metrics['schedule_delay_p50'],
                        'Schedule_Delay_P90': metrics['schedule_delay_p90'],
                        'Schedule_Delay_P99': metrics['schedule_delay_p99'],
                        'E2E_Latency_Avg': metrics['e2e_latency_avg'],
                        'E2E_Latency_P50': metrics['e2e_latency_p50'],
                        'E2E_Latency_P90': metrics['e2e_latency_p90'],
                        'E2E_Latency_P99': metrics['e2e_latency_p99'],
                        'Throughput': metrics['throughput'],
                        'Total_Requests': metrics['total_requests'],
                        'Total_Input_Tokens': metrics['total_input_tokens'],
                        'Total_Output_Tokens': metrics['total_output_tokens'],
                        'Process_Time_Seconds': timing_data['process_time'],
                        'Total_Time_With_Init_Seconds': timing_data['total_time_with_init']
                    })

        return pd.DataFrame(results_list)

def main():
    base_dir = "/home/ec2-user/s3-local/vidur_qps/vidur-simulator/benchmarks/llm/vllm/latency/vllm_output/vidur_qps/qu_brand"
    output_dir = "./vllm_bench_results/qu_brand/qps_search"
    
    parser = VLLMBenchQu(base_dir, output_dir)
    results_df = parser.parse_all()
    
    if results_df.empty:
        print("No results found!")
        return
    
    # Save results
    csv_path, summary_path = parser.save_results(results_df)
    
    print(f"Results saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")
    
    print(f"\nFound {len(results_df)} benchmark results")
    print("\nSample results:")
    print(results_df[['Device', 'QPS_Dir', 'Model', 'Tensor_Parallel', 'First_Token_P99', 'Inter_Token_P99']].head(10))

if __name__ == "__main__":
    main()