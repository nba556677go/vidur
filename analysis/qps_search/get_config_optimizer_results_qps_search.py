
import os
import json
import pandas as pd
import numpy as np
import csv
import re

def get_device_costs():
    """
    Parse the unitcost.csv to get hourly costs for different device types
    and include GPUs per node information
    """
    device_costs = {
        'p4d_a100_40g_nvlink': {'cost': None, 'gpus_per_node': 8},  # p4d.24xlarge has 8 A100 GPUs
        'h100_p5': {'cost': None, 'gpus_per_node': 8},  # p5.48xlarge has 8 H100 GPUs
        'l40s_g6e48': {'cost': None, 'gpus_per_node': 8},  # g6e.48xlarge has 8 L40S GPUs
        'a10g_g5': {'cost': None, 'gpus_per_node': 8},  # g5.48xlarge has 8 A10G GPUs
        'l4_g6': {'cost': None, 'gpus_per_node': 8},  # g6.48xlarge has 8 L4 GPUs
    }
    
    # Instance type to device mapping
    instance_to_device = {
        'p4d.24xlarge': 'p4d_a100_40g_nvlink',
        'p5.48xlarge': 'h100_p5',
        'g6e.48xlarge': 'l40s_g6e48',
        'g5.48xlarge': 'a10g_g5',
        'g6.48xlarge': 'l4_g6'
    }
    
    csv_path = "../unitcost.csv"
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                instance_type = row.get('instance', '').strip()
                
                if instance_type in instance_to_device:
                    device = instance_to_device[instance_type]
                    
                    try:
                        cost = float(row.get('cost', '0').strip())
                        device_costs[device]['cost'] = cost
                        print(f'Found cost for {device} ({instance_type}): ${cost:.2f}')
                    except (ValueError, TypeError) as e:
                        raise ValueError(f'Error parsing cost for {instance_type}: {e}')
    except FileNotFoundError:
        raise FileNotFoundError(f"Cost file not found: {csv_path}")
    
    # Check for missing costs and raise error if any are not found
    missing_costs = [device for device, info in device_costs.items() if info['cost'] is None]
    if missing_costs:
        raise ValueError(f"Cost not found for devices: {missing_costs}")
    
    print("Device information:")
    for device, info in device_costs.items():
        print(f"  {device}: ${info['cost']:.2f} per hour, {info['gpus_per_node']} GPUs per node")
        
    return device_costs

# Directory containing the optimizer output
#CONFIG_DIR = "/home/ec2-user/vidur-simulator/config_optimizer_output_r8_r16/runs"
CONFIG_DIRS = [
    "/home/ec2-user/vidur-simulator/simulator_output/qu_brand/qps_search/config_optimizer_output_qwen1.5_r1_r2_r4_r8_r16_a10g_g5/runs",
    "/home/ec2-user/vidur-simulator/simulator_output/qu_brand/qps_search/config_optimizer_output_qwen1.5_r1_r2_r4_r8_r16_l4_g6/runs",
    "/home/ec2-user/vidur-simulator/simulator_output/qu_brand/qps_search/config_optimizer_output_qwen1.5_r1_r2_r4_r8_r16_a100_p4d/runs",
    "/home/ec2-user/vidur-simulator/simulator_output/qu_brand/qps_search/config_optimizer_output_qwen1.5_r1_r2_r4_r8_r16_h100_p5/runs",
    "/home/ec2-user/vidur-simulator/simulator_output/qu_brand/qps_search/config_optimizer_output_qwen1.5_r1_r2_r4_r8_r16_l40s_g6e48/runs",
    "/home/ec2-user/vidur-simulator/simulator_output/qu_brand/qps_search/config_optimizer_output_llama_8b_r1_r2_r4_r8_r16_a10g_g5/runs",
    "/home/ec2-user/vidur-simulator/simulator_output/qu_brand/qps_search/config_optimizer_output_llama_8b_r1_r2_r4_r8_r16_l4_g6/runs",
    "/home/ec2-user/vidur-simulator/simulator_output/qu_brand/qps_search/config_optimizer_output_llama_8b_r1_r2_r4_r8_r16_a100_p4d/runs",
    "/home/ec2-user/vidur-simulator/simulator_output/qu_brand/qps_search/config_optimizer_output_llama_8b_r1_r2_r4_r8_r16_h100_p5/runs",
     "/home/ec2-user/vidur-simulator/simulator_output/qu_brand/qps_search/config_optimizer_output_llama_8b_r1_r2_r4_r8_r16_l40s_g6e48/runs",
   

]
# Data structure to hold results
results = []

# Get device costs
device_costs = get_device_costs()

# SLO limit example
slo_limit = 0.25  # 200ms example for TTFT
exec_slo = 7.8  # slo for total execution time
inter_token_slo = 0.015  # 8ms in seconds

# Walk through all config directories
for config_dir in CONFIG_DIRS:
    base_dir = os.path.expanduser(config_dir)

    # Walk through all run directories in each config dir
    for run_dir in os.listdir(base_dir):
        run_path = os.path.join(base_dir, run_dir)
        if not os.path.isdir(run_path):
            continue
        
        # Process each QPS directory
        for qps_dir in os.listdir(run_path):
            if not re.match(r'^r\d+_q', qps_dir):
                continue
                
            # Extract QPS value from directory name
            try:
                qps = float(qps_dir.split('_q')[1])
            except:
                continue
                
            qps_path = os.path.join(run_path, qps_dir)
            
            # Find the timestamped directory
            timestamp_dirs = [d for d in os.listdir(qps_path) if os.path.isdir(os.path.join(qps_path, d))]
            if not timestamp_dirs:
                continue
                
            # Use the first timestamped directory
            timestamp_path = os.path.join(qps_path, timestamp_dirs[0])
            
            # Check if config.json and request_metrics.csv exist
            config_path = os.path.join(timestamp_path, "config.json")
            metrics_path = os.path.join(timestamp_path, "request_metrics.csv")
            
            if not (os.path.exists(config_path) and os.path.exists(metrics_path)):
                continue
                
            # Parse config
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Extract relevant config details
            replica_config = config.get('cluster_config', {}).get('replica_config', {})
            num_replicas = config.get('cluster_config', {}).get('num_replicas', 0)
            tensor_parallel_size = replica_config.get('tensor_parallel_size', 0)
            num_pipeline_stages = replica_config.get('num_pipeline_stages', 0)
            device = replica_config.get('device', 'Unknown')
            network_device = replica_config.get('network_device', 'Unknown')
            
            # Extract token lengths (prefill and decode)
            request_generator_config = config.get('request_generator_config', {})
            length_generator_config = request_generator_config.get('length_generator_config', {})
            prefill_tokens = length_generator_config.get('prefill_tokens', 0)
            decode_tokens = length_generator_config.get('decode_tokens', 0)
            
            # Get scheduler info and related parameters
            scheduler_config = config.get('scheduler_config', {})
            replica_scheduler_config = config.get('replica_scheduler_config', {})
            cluster_config = config.get('cluster_config', {})
            
            # Look for scheduler_type - need to check in several possible locations
            scheduler_type = None
            
            # 1. Check in scheduler_config
            if scheduler_type is None:
                scheduler_type = scheduler_config.get('scheduler_type', None)
                
            # 2. Check in cluster_config -> replica_scheduler_config -> name
            if scheduler_type is None and 'replica_scheduler_config' in cluster_config:
                scheduler_type = cluster_config['replica_scheduler_config'].get('name', None)
                
            # 3. Check in global_scheduler_config -> name
            if scheduler_type is None and 'global_scheduler_config' in cluster_config:
                scheduler_type = cluster_config['global_scheduler_config'].get('name', None)
                
            # Default to Unknown if not found
            if scheduler_type is None:
                scheduler_type = "Unknown"
            
            # Get chunk_size - need to check in several possible locations
            chunk_size = None
            
            # 1. Try cluster_config -> replica_scheduler_config
            if chunk_size is None and 'replica_scheduler_config' in cluster_config:
                chunk_size = cluster_config['replica_scheduler_config'].get('chunk_size', None)
            
            # 2. Try replica_scheduler_config directly (may be at top level)
            if chunk_size is None:
                chunk_size = replica_scheduler_config.get('chunk_size', None)
                
            # 3. Try scheduler_config -> replica_scheduler_config
            if chunk_size is None:
                scheduler_replica_config = scheduler_config.get('replica_scheduler_config', {})
                chunk_size = scheduler_replica_config.get('chunk_size', None)
                
            # 4. Fall back to sarathi_chunk_size if needed
            if chunk_size is None:
                chunk_size = scheduler_config.get('sarathi_chunk_size', None)
                
            # Get batch size in similar way
            sarathi_batch_size = None
            
            # 1. Try directly in scheduler_config
            if sarathi_batch_size is None:
                sarathi_batch_size = scheduler_config.get('batch_size', None)
                
            # 2. Try in replica_scheduler_config
            if sarathi_batch_size is None and 'replica_scheduler_config' in cluster_config:
                sarathi_batch_size = cluster_config['replica_scheduler_config'].get('batch_size_cap', None)
            # Get model name
            model_name = replica_config.get('model_name', 'Unknown')

            # Read metrics
            metrics_df = pd.read_csv(metrics_path)
            
            # Calculate total runtime: max(request_arrived_at + request_e2e_time)
            if 'request_arrived_at' in metrics_df.columns and 'request_e2e_time' in metrics_df.columns:
                total_runtime_seconds = (metrics_df['request_arrived_at'] + metrics_df['request_e2e_time']).max()
            else:
                # Fallback: use max request_e2e_time if columns are missing
                total_runtime_seconds = metrics_df['request_e2e_time'].max() if 'request_e2e_time' in metrics_df.columns else 0
            
            # Convert to hours
            total_runtime_hours = total_runtime_seconds / 3600.0
            
            # Calculate P99 of prefill_e2e_time
            p99_ttft = metrics_df['prefill_e2e_time'].quantile(0.99)
            
            # Calculate P99 and P50 of total request execution time from request_execution_time column
            p99_exec_time = None
            p50_exec_time = None
            if 'request_execution_time' in metrics_df.columns:
                p99_exec_time = metrics_df['request_execution_time'].quantile(0.99)
                p50_exec_time = metrics_df['request_execution_time'].quantile(0.50)
            
            # Calculate P99 of decode_time_execution_plus_preemption_normalized
            p99_inter_token_latency = None
            if 'decode_time_execution_plus_preemption_normalized' in metrics_df.columns:
                p99_inter_token_latency = metrics_df['decode_time_execution_plus_preemption_normalized'].quantile(0.99)
            
            # Store the result
            results.append({
                'run_id': run_dir,
                'qps': qps,
                'p99_ttft': p99_ttft,
                'p99_exec_time': p99_exec_time,
                'p50_exec_time': p50_exec_time,
                'p99_inter_token_latency': p99_inter_token_latency,
                'num_replicas': num_replicas,
                'tensor_parallel_size': tensor_parallel_size,
                'num_pipeline_stages': num_pipeline_stages,
                'device': device,
                'network_device': network_device,
                'model_name': model_name,
                'scheduler_type': scheduler_type,
                'chunk_size': chunk_size,
                'batch_size': sarathi_batch_size,
                'prefill_tokens': prefill_tokens,
                'decode_tokens': decode_tokens,
                'total_runtime_hours': total_runtime_hours,
                'config_path': config_path,
                'metrics_path': metrics_path
            })

# Convert results to DataFrame
df = pd.DataFrame(results)

if len(df) == 0:
    print("No data found!")
else:
    # Save the raw data
    
    
    # Calculate QPS per dollar using the actual device costs and proper device count
    df['device_cost_per_hour'] = df['network_device'].apply(lambda x: device_costs.get(x, {}).get('cost', 0))
    df['gpus_per_node'] = df['network_device'].apply(lambda x: device_costs.get(x, {}).get('gpus_per_node', 8))
    
    # Calculate replicas that can fit on one node
    df['replica_per_node'] = df.apply(
        lambda x: x['gpus_per_node'] / (x['tensor_parallel_size'] * x['num_pipeline_stages']), 
        axis=1
    )
    assert all(df['replica_per_node'] > 0), "Replica per node must be greater than 0"    
    
    # Calculate number of nodes needed
    df['nodes_needed'] = df.apply(
        lambda x: np.ceil(x['num_replicas'] / x['replica_per_node']),
        axis=1
    )
    
    # Calculate total cost per hour
    df['total_cost_per_hour'] = df['device_cost_per_hour'] * df['nodes_needed']
    
    # Calculate total cost for the entire run
    df['total_cost'] = df['total_cost_per_hour'] * df['total_runtime_hours']
    
    # Calculate QPS per dollar using total cost
    df['qps_per_dollar'] = df.apply(
        lambda x: x['qps'] / x['total_cost'] if x['total_cost'] > 0 else 0, 
        axis=1
    )
    
    # Save all columns including the calculated ones
    df.to_csv("config_optimizer_results.csv", index=False)
    
    print("Analysis complete.")
    print(f"Raw data saved: config_optimizer_results.csv")
    print(f"Total configurations processed: {len(df)}")

