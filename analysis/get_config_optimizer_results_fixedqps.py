
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import csv
import re

def get_device_costs(csv_path):
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
   "/home/ec2-user/vidur-simulator/simulator_output/qu_brand/compute_a100_p4d/network_p4d_a100_40g_nvlink",
   "/home/ec2-user/vidur-simulator/simulator_output/qu_brand/compute_h100_p5/network_h100_p5",
   "/home/ec2-user/vidur-simulator/simulator_output/qu_brand/compute_l40s_g6e48/network_l40s_g6e48",
   "/home/ec2-user/vidur-simulator/simulator_output/qu_brand/compute_a10g_g5/network_a10g_g5",
   "/home/ec2-user/vidur-simulator/simulator_output/qu_brand/compute_l4_g6/network_l4_g6"
]

# Output directory for saving results
output_dir = "./vidur_results/qu_brand/fixed_qps"
os.makedirs(output_dir, exist_ok=True)
# Data structure to hold results
results = []

# Get device costs
device_costs = get_device_costs(csv_path = "./unitcost.csv")

# SLO limit example
slo_limit = 0.25  # 200ms example for TTFT
exec_slo = 7.8  # slo for total execution time
inter_token_slo = 0.015  # 8ms in seconds

# Walk through all config directories
for config_dir in CONFIG_DIRS:
    base_dir = os.path.expanduser(config_dir)

    # Walk through model directories (e.g., meta-llama, Qwen)
    for model_org in os.listdir(base_dir):
        model_org_path = os.path.join(base_dir, model_org)
        if not os.path.isdir(model_org_path):
            continue
            
        # Walk through specific model directories (e.g., Meta-Llama-3-8B)
        for model_name in os.listdir(model_org_path):
            model_path = os.path.join(model_org_path, model_name)
            if not os.path.isdir(model_path):
                continue
        
            # Process each QPS directory
            for qps_dir in os.listdir(model_path):
                if not re.match(r'^qps\d+', qps_dir):
                    continue

                    
                # Extract QPS value from directory name
                try:
                    qps = float(qps_dir.replace('qps', ''))
                except:
                    continue
                    
                qps_path = os.path.join(model_path, qps_dir)
                
                # Process all timestamp directories that have both files
                timestamp_dirs = [d for d in os.listdir(qps_path) if os.path.isdir(os.path.join(qps_path, d))]

                if not timestamp_dirs:
                    continue
                    
                # Process each timestamp directory that has both files
                for timestamp_dir in timestamp_dirs:
                    timestamp_path = os.path.join(qps_path, timestamp_dir)
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
                    model_name_from_config = replica_config.get('model_name', 'Unknown')

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
                        'run_id': f"{model_org}/{model_name}/{timestamp_dir}",
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
                        'model_name': model_name_from_config,
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
    
    
    # Display top 5 results sorted by p99_ttft
    print("Top 5 configs by lowest P99 TTFT:")
    top_configs = df.sort_values('p99_ttft').head(5)
    print(top_configs)
    
    # Create a unique color map for the different configurations
    unique_configs = {}
    
    # Use network_device for coloring
    unique_network_device = df['network_device'].unique()
    
    for i, val in enumerate(unique_network_device):
        unique_configs[val] = i
    
    # Create custom colormap
    colors = plt.cm.viridis(np.linspace(0, 1, max(len(unique_configs), 1)))
    
    # Calculate QPS per dollar using the actual device costs and proper device count
    # Get cost per hour and GPUs per node for each device
    df['device_cost_per_hour'] = df['network_device'].apply(lambda x: device_costs.get(x, {}).get('cost', 0))
    df['gpus_per_node'] = df['network_device'].apply(lambda x: device_costs.get(x, {}).get('gpus_per_node', 8))
    
    # Calculate how many replicas can fit on one node based on device GPU count
    # Add comments to explain the data flow
    # These columns are calculated on-the-fly and not persisted in the DataFrame
    # To save them, we need to store them before saving to CSV
    
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
    df.to_csv(os.path.join(output_dir, "config_optimizer_results.csv"), index=False)
    # Calculate best configs for different metrics
    slo_compliant = df[(df['p99_ttft'] <= slo_limit) & (df['p99_exec_time'] <= exec_slo)]
    
    # Best config for QPS (max QPS under SLO)
    if len(slo_compliant) > 0:
        best_config_qps = slo_compliant.loc[slo_compliant['qps'].idxmax()]
    else:
        print("Warning: no config under SLO configured for QPS, falling back to min p99_ttft...")
        best_config_qps = df.sort_values('p99_ttft').iloc[0]
    
    # Best config for QPS per dollar (max QPS per dollar under SLO)
    if len(slo_compliant) > 0:
        best_config_qps_per_dollar = slo_compliant.loc[slo_compliant['qps_per_dollar'].idxmax()]
    else:
        print("Warning: no config under SLO configured for QPS per dollar, falling back to min p99_ttft...")
        best_config_qps_per_dollar = df.sort_values('p99_ttft').iloc[0]
    
    # For the third subplot, use the QPS best config
    best_config_third_plot = best_config_qps
    
    # Check if we have execution time data
    has_exec_time = all(pd.notna(df['p99_exec_time'])) if len(df) > 0 else False
    
    # Create the best config descriptions
    best_desc_qps = (f"Best QPS Config: PP={best_config_qps['num_pipeline_stages']}, "
                    f"TP={best_config_qps['tensor_parallel_size']}, "
                    f"Replicas={best_config_qps['num_replicas']}, "
                    f"Nodes={best_config_qps['nodes_needed']}, "
                    f"Scheduler={best_config_qps['scheduler_type']}, "
                    f"Chunk={best_config_qps['chunk_size']}, "
                    f"Batch={best_config_qps['batch_size']}, "
                    f"SKU={best_config_qps['device']}, "
                    f"QPS = {best_config_qps['qps']:.2f}")
    
    best_desc_qps_per_dollar = (f"Best QPS/Dollar Config: PP={best_config_qps_per_dollar['num_pipeline_stages']}, "
                               f"TP={best_config_qps_per_dollar['tensor_parallel_size']}, "
                               f"Replicas={best_config_qps_per_dollar['num_replicas']}, "
                               f"Nodes={best_config_qps_per_dollar['nodes_needed']}, "
                               f"Scheduler={best_config_qps_per_dollar['scheduler_type']}, "
                               f"Chunk={best_config_qps_per_dollar['chunk_size']}, "
                               f"Batch={best_config_qps_per_dollar['batch_size']}, "
                               f"SKU={best_config_qps_per_dollar['device']}, "
                               f"QPS = {best_config_qps['qps']:.2f}, "
                               f"QPS/$ = {best_config_qps_per_dollar['qps_per_dollar']:.4f}")

    # =========================================
    # Figure 1: QPS Scatter Plot (4 subplots - including p99_inter_token_latency)
    # =========================================
    fig1, axes1 = plt.subplots(2, 2, figsize=(18, 16), gridspec_kw={'width_ratios': [1, 1.2], 'height_ratios': [1, 1]})
    #fig1.suptitle("LLM Performance: QPS Analysis", fontsize=21)
    fig1.text(0.5, 0.97, best_desc_qps, ha='center', fontsize=17)

    # Subplot 1: QPS vs P99 TTFT
    ax = axes1[0, 0]
    ax.set_title("QPS vs P99 Time to First Token", fontsize=19)
    
    for network_device in unique_network_device:
        subset = df[df['network_device'] == network_device]
        ax.scatter(subset['p99_ttft'], subset['qps'], 
                   label=network_device,
                   color=colors[unique_configs[network_device]],
                   s=80, alpha=0.7)
    
    # No star for first subplot
    
    # Add SLO limit line
    ax.axvline(x=slo_limit, color='red', linestyle='--', label='SLO Limit (200ms)')
    ax.axvspan(0, slo_limit, alpha=0.1, color='green', label='SLO Compliant Region')
    
    ax.set_xlabel("Time to First Token - P99 (s)", fontsize=17)
    ax.set_ylabel("QPS", fontsize=17)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Configuration", fontsize=13)
    
    # Subplot 2: QPS vs P99 Request Total Latency
    ax = axes1[0, 1]
    ax.set_title("QPS vs P99 Request Total Latency", fontsize=19)
    
    if has_exec_time:
        for network_device in unique_network_device:
            subset = df[df['network_device'] == network_device]
            ax.scatter(subset['p99_exec_time'], subset['qps'], 
                      label=network_device,
                      color=colors[unique_configs[network_device]],
                      s=80, alpha=0.7)
        
        # No star for second subplot
        
        # Add SLO limit line
        ax.axvline(x=exec_slo, color='red', linestyle='--', label=f'SLO Limit ({exec_slo}s)')
        ax.axvspan(0, exec_slo, alpha=0.1, color='green', label='SLO Compliant Region')
        
        ax.set_xlabel("Request Execution Time - P99 (s)", fontsize=17)
        ax.set_ylabel("QPS", fontsize=17)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Configuration", fontsize=13)
    else:
        ax.text(0.5, 0.5, "P99 Request Total Latency metrics not available", 
                ha='center', va='center', fontsize=19)
    
    # Subplot 3: QPS vs P99 Inter-Token Latency
    ax = axes1[1, 0]
    ax.set_title("QPS vs P99 Inter-Token Latency", fontsize=19)
    
    # Check if we have inter-token latency data
    has_inter_token_latency = pd.notna(df['p99_inter_token_latency']).any()
    
    if has_inter_token_latency:
        for network_device in unique_network_device:
            subset = df[df['network_device'] == network_device]
            ax.scatter(subset['p99_inter_token_latency'], subset['qps'], 
                     label=network_device,
                     color=colors[unique_configs[network_device]],
                     s=80, alpha=0.7)
        
        # No star for third subplot
        
        # Add SLO limit line for inter-token latency (8ms)
        ax.axvline(x=inter_token_slo, color='red', linestyle='--', label=f'Inter-Token SLO ({inter_token_slo*1000}ms)')
        ax.axvspan(0, inter_token_slo, alpha=0.1, color='green', label='SLO Compliant Region')
        
        ax.set_xlabel("Inter-Token Latency - P99 (s)", fontsize=17)
        ax.set_ylabel("QPS", fontsize=17)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Configuration", fontsize=13)
    else:
        ax.text(0.5, 0.5, "P99 Inter-Token Latency metrics not available", 
               ha='center', va='center', fontsize=19)
    
    # Subplot 4: P99 Inter-Token Latency vs P99 TTFT, colored by QPS
    ax = axes1[1, 1]
    ax.set_title("P99 Inter-Token Latency vs P99 TTFT (Colored by QPS)", fontsize=19)
    
    # Check if we have inter-token latency data
    has_inter_token_latency = pd.notna(df['p99_inter_token_latency']).any()
    
    if has_inter_token_latency:
        # Create scatter plot with QPS as color
        scatter = ax.scatter(df['p99_ttft'], df['p99_inter_token_latency'], 
                           c=df['qps'], cmap='viridis', s=100, alpha=0.7)
        
        # Add colorbar with proper spacing
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(scatter, cax=cax)
        cbar.set_label('QPS', fontsize=17)
        
        # Highlight the best config (only on fourth subplot)
        ax.scatter(best_config_third_plot['p99_ttft'], best_config_third_plot['p99_inter_token_latency'], 
                  color='gold', s=200, marker='*', 
                  label=f"Best: {best_config_third_plot['device']} TP{best_config_third_plot['tensor_parallel_size']}/PP{best_config_third_plot['num_pipeline_stages']}", 
                  edgecolor='black', zorder=10)
        
        # Add SLO limit lines
        ax.axvline(x=slo_limit, color='red', linestyle='--', label=f'TTFT SLO ({slo_limit*1000}ms)')
        ax.axhline(y=inter_token_slo, color='red', linestyle=':', label=f'Inter-Token SLO ({inter_token_slo*1000}ms)')
        
        # Add SLO compliant region (bottom-left rectangle)
        ax.fill_between([0, slo_limit], 0, inter_token_slo, alpha=0.1, color='green', label='SLO Compliant Region')
        
        ax.set_xlabel("Time to First Token - P99 (s)", fontsize=17)
        ax.set_ylabel("Inter-Token Latency - P99 (s)", fontsize=17)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=13)
    else:
        ax.text(0.5, 0.5, "P99 Inter-Token Latency metrics not available", 
                ha='center', va='center', fontsize=19)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "qps_scatter.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # =================================================
    # Figure 2: QPS per Dollar Scatter Plot (4 subplots - including p99_inter_token_latency)
    # =================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(18, 16), gridspec_kw={'width_ratios': [1, 1.2], 'height_ratios': [1, 1]})
    #fig2.suptitle("LLM Cost Efficiency: QPS per Dollar Analysis", fontsize=21)
    fig2.text(0.5, 0.97, best_desc_qps_per_dollar, ha='center', fontsize=17)
    
    # Subplot 1: QPS per Dollar vs P99 TTFT
    ax = axes2[0, 0]
    ax.set_title("QPS per Dollar vs P99 TTFT", fontsize=19)
    
    for network_device in unique_network_device:
        subset = df[df['network_device'] == network_device]
        ax.scatter(subset['p99_ttft'], subset['qps_per_dollar'], 
                   label=network_device,
                   color=colors[unique_configs[network_device]],
                   s=80, alpha=0.7)
    
    # No star for first subplot
    
    # Add SLO limit line
    ax.axvline(x=slo_limit, color='red', linestyle='--', label=f'SLO Limit ({slo_limit*1000}ms)')
    ax.axvspan(0, slo_limit, alpha=0.1, color='green', label='SLO Compliant Region')
    
    ax.set_xlabel("Time to First Token - P99 (s)", fontsize=17)
    ax.set_ylabel("QPS per Dollar", fontsize=17)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Configuration", fontsize=13)
    
    # Subplot 2: QPS per Dollar vs P99 Request Total Latency
    ax = axes2[0, 1]
    ax.set_title("QPS per Dollar vs P99 Request Latency", fontsize=19)
    
    if has_exec_time:
        for network_device in unique_network_device:
            subset = df[df['network_device'] == network_device]
            ax.scatter(subset['p99_exec_time'], subset['qps_per_dollar'], 
                      label=network_device,
                      color=colors[unique_configs[network_device]],
                      s=80, alpha=0.7)
        
        # No star for second subplot
        
        # Add SLO limit line
        ax.axvline(x=exec_slo, color='red', linestyle='--', label='SLO Limit (5s)')
        ax.axvspan(0, exec_slo, alpha=0.1, color='green', label='SLO Compliant Region')
        
        ax.set_xlabel("Request Execution Time - P99 (s)", fontsize=17)
        ax.set_ylabel("QPS per Dollar", fontsize=17)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Configuration", fontsize=13)
    else:
        ax.text(0.5, 0.5, "P99 Request Total Latency metrics not available", 
                ha='center', va='center', fontsize=19)
    
    # Subplot 3: QPS per Dollar vs P99 Inter-Token Latency
    ax = axes2[1, 0]
    ax.set_title("QPS per Dollar vs P99 Inter-Token Latency", fontsize=19)
    
    # Check if we have inter-token latency data
    has_inter_token_latency = pd.notna(df['p99_inter_token_latency']).any()
    
    if has_inter_token_latency:
        for network_device in unique_network_device:
            subset = df[df['network_device'] == network_device]
            ax.scatter(subset['p99_inter_token_latency'], subset['qps_per_dollar'], 
                     label=network_device,
                     color=colors[unique_configs[network_device]],
                     s=80, alpha=0.7)
        
        # No star for third subplot
        
        # Add a reasonable SLO limit line for inter-token latency (e.g., 20ms)
        ax.axvline(x=inter_token_slo, color='red', linestyle='--', label=f'Inter-Token SLO ({inter_token_slo*1000}ms)')
        ax.axvspan(0, inter_token_slo, alpha=0.1, color='green', label='SLO Compliant Region')
        
        ax.set_xlabel("Inter-Token Latency - P99 (s)", fontsize=17)
        ax.set_ylabel("QPS per Dollar", fontsize=17)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Configuration", fontsize=13)
    else:
        ax.text(0.5, 0.5, "P99 Inter-Token Latency metrics not available", 
               ha='center', va='center', fontsize=19)
    
    # Subplot 4: P99 Inter-Token Latency vs P99 TTFT, colored by QPS per Dollar
    ax = axes2[1, 1]
    ax.set_title("P99 Inter-Token Latency vs P99 TTFT (Colored by QPS/$)", fontsize=19)
    
    # Check if we have inter-token latency data
    has_inter_token_latency = pd.notna(df['p99_inter_token_latency']).any()
    
    if has_inter_token_latency:
        # Create scatter plot with QPS per dollar as color
        scatter = ax.scatter(df['p99_ttft'], df['p99_inter_token_latency'], 
                           c=df['qps_per_dollar'], cmap='viridis', s=100, alpha=0.7)
        
        # Add colorbar with proper spacing
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(scatter, cax=cax)
        cbar.set_label('QPS per Dollar', fontsize=17)
        
        # Highlight the best config (only on fourth subplot)
        ax.scatter(best_config_third_plot['p99_ttft'], best_config_third_plot['p99_inter_token_latency'], 
                  color='gold', s=250, marker='*', 
                  label=f"Best: {best_config_third_plot['device']} TP{best_config_third_plot['tensor_parallel_size']}/PP{best_config_third_plot['num_pipeline_stages']}", 
                  edgecolor='black', zorder=10)
        
        # Add SLO limit lines
        ax.axvline(x=slo_limit, color='red', linestyle='--', label=f'TTFT SLO ({slo_limit*1000}ms)')
        ax.axhline(y=inter_token_slo, color='red', linestyle=':', label=f'Inter-Token SLO ({inter_token_slo*1000}ms)')
        
        # Add SLO compliant region (bottom-left rectangle)
        ax.fill_between([0, slo_limit], 0, inter_token_slo, alpha=0.1, color='green', label='SLO Compliant Region')
        
        ax.set_xlabel("Time to First Token - P99 (s)", fontsize=17)
        ax.set_ylabel("Inter-Token Latency - P99 (s)", fontsize=17)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=13)
    else:
        ax.text(0.5, 0.5, "P99 Inter-Token Latency metrics not available", 
                ha='center', va='center', fontsize=19)
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "qps_per_dollar_scatter.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Add parallelism strategies plotting with vLLM benchmark comparison
    def load_vllm_benchmark_data(instance_type):
        """Load vLLM benchmark data for comparison."""
        vllm_data = {}
        
        # Map instance types to vLLM benchmark paths
        vllm_paths = {
            'p4d_a100_40g_nvlink': [
                "/home/ec2-user/vidur-simulator/analysis/vllm_bench_results/qu_brand/fixed_qps/a100_p4d/nprompt300"
            ],
            'a10g_g5': [
                "/home/ec2-user/vidur-simulator/analysis/vllm_bench_results/qu_brand/fixed_qps/a10g_g5/nprompt300"
            ],
            'h100_p5': [
                "/home/ec2-user/vidur-simulator/analysis/vllm_bench_results/qu_brand/fixed_qps/h100_p5/nprompt300"
            ],
            'l4_g6': [
                "/home/ec2-user/vidur-simulator/analysis/vllm_bench_results/qu_brand/fixed_qps/l4_g6/nprompt300"
            ],
            'l40s_g6e48': [
                "/home/ec2-user/vidur-simulator/analysis/vllm_bench_results/qu_brand/fixed_qps/l40s_g6e48/nprompt300"
            ]
        }
        
        print(f"Debug: Looking for vLLM data for {instance_type}")
        print(f"Debug: Paths to check: {vllm_paths.get(instance_type, [])}")
        
        if instance_type not in vllm_paths:
            return vllm_data
            
        for path in vllm_paths[instance_type]:
            print(f"Debug: Checking path: {path}")
            print(f"Debug: Path exists: {os.path.exists(path)}")
            try:
                if path.endswith('.csv'):
                    # Process time CSV files
                    if os.path.exists(path):
                        print(f"Debug: Loading CSV file: {path}")
                        vllm_df = pd.read_csv(path)
                        print(f"Debug: CSV loaded with {len(vllm_df)} rows")
                        for _, row in vllm_df.iterrows():
                            model = row['model']
                            qps = row['qps']
                            tp = row['tensor_parallel_size']
                            pp = row['pipeline_parallel_size'] if 'pipeline_parallel_size' in row else 1
                            dp = row['data_parallel_size'] if 'data_parallel_size' in row else 1
                            process_time = row['total_process_time_seconds']
                            
                            # Calculate cost using process time
                            device_cost_per_hour = device_costs.get(instance_type, {}).get('cost', 0)
                            gpus_per_node = device_costs.get(instance_type, {}).get('gpus_per_node', 8)
                            
                            # Calculate replicas and nodes needed
                            total_replicas = tp * pp * dp
                            replica_per_node = gpus_per_node / (tp * pp)
                            nodes_needed = np.ceil(total_replicas / replica_per_node)
                            
                            total_cost_per_hour = device_cost_per_hour * nodes_needed
                            total_cost = total_cost_per_hour * (process_time / 3600.0)
                            
                            key = (model, qps, total_replicas, tp)
                            if key not in vllm_data or vllm_data[key] > total_cost:
                                vllm_data[key] = total_cost
                                print(f"Debug: Added vLLM entry: {key} -> ${total_cost:.4f}")
                else:
                    # Directory with QPS subdirectories
                    if os.path.exists(path):
                        print(f"Debug: Loading directory: {path}")
                        qps_dirs = os.listdir(path)
                        print(f"Debug: Found QPS directories: {qps_dirs}")
                        for qps_dir in qps_dirs:
                            if qps_dir.startswith('qps'):
                                qps_value = float(qps_dir.replace('qps', ''))
                                summary_path = os.path.join(path, qps_dir)
                                
                                # Find summary CSV files
                                for file in os.listdir(summary_path):
                                    if file.startswith('summary_') and file.endswith('.csv'):
                                        summary_file = os.path.join(summary_path, file)
                                        summary_df = pd.read_csv(summary_file)
                                        
                                        for _, row in summary_df.iterrows():
                                            model = row['Model']
                                            tp = int(row['Tensor_Parallel'])
                                            pp = int(row['Pipeline_Parallel']) if 'Pipeline_Parallel' in row else 1
                                            replicas = int(row['Num_Replicas'])
                                            process_time = row['Process_Time_Seconds']
                                            
                                            # Calculate cost
                                            device_cost_per_hour = device_costs.get(instance_type, {}).get('cost', 0)
                                            gpus_per_node = device_costs.get(instance_type, {}).get('gpus_per_node', 8)
                                            
                                            replica_per_node = gpus_per_node / (tp * pp)
                                            nodes_needed = np.ceil(replicas / replica_per_node)
                                            
                                            total_cost_per_hour = device_cost_per_hour * nodes_needed
                                            total_cost = total_cost_per_hour * (process_time / 3600.0)
                                            
                                            key = (model, qps_value, replicas, tp)
                                            if key not in vllm_data or vllm_data[key] > total_cost:
                                                vllm_data[key] = total_cost
                                                print(f"Debug: Added vLLM summary entry: {key} -> ${total_cost:.4f}")
            except Exception as e:
                print(f"Error loading vLLM data from {path}: {e}")
                continue
                
        return vllm_data
    
    def plot_parallelism_strategies(instance_type='p4d_a100_40g_nvlink'):
        """Plot parallelism strategies for each QPS value for a specific instance type with nodes_needed==1."""
        # Create parallel_figs directory
        parallel_figs_dir = os.path.join(output_dir, 'parallel_figs')
        os.makedirs(parallel_figs_dir, exist_ok=True)
        
        # Load vLLM benchmark data
        vllm_data = load_vllm_benchmark_data(instance_type)
        print(f"Debug: Loaded {len(vllm_data)} vLLM benchmark entries for {instance_type}")
        if len(vllm_data) > 0:
            print(f"Debug: Sample vLLM entries: {list(vllm_data.items())[:3]}")
        
        # Filter for specific instance and nodes_needed==1
        instance_data = df[
            (df['network_device'] == instance_type) & 
            (df['nodes_needed'] == 1)
        ].copy()
        
        if len(instance_data) == 0:
            print(f"No data found for {instance_type} with nodes_needed==1")
            return
        
        # Get unique QPS values and models
        unique_qps = sorted(instance_data['qps'].unique())
        unique_models = instance_data['model_name'].unique()
        
        # Create separate plot for each QPS value
        for qps_value in unique_qps:
            qps_data = instance_data[instance_data['qps'] == qps_value]
            
            if len(qps_data) == 0:
                continue
                
            n_models = len(unique_models)
            
            # Create subplots for each model
            if n_models == 1:
                fig, axes = plt.subplots(1, 1, figsize=(14, 8))
                axes = [axes]
            else:
                n_cols = int(np.ceil(np.sqrt(n_models)))
                n_rows = int(np.ceil(n_models / n_cols))
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(10*n_cols, 6*n_rows))
                axes = axes.flatten() if n_models > 1 else [axes]
            
            for i, model_name in enumerate(unique_models):
                ax = axes[i]
                model_data = qps_data[qps_data['model_name'] == model_name]
                
                if len(model_data) == 0:
                    ax.text(0.5, 0.5, f"No data for {model_name} at QPS {qps_value}", ha='center', va='center')
                    ax.set_title(f'{model_name} - {instance_type} - QPS {qps_value}')
                    continue
                
                # Group by parallelism strategy and get min total cost
                config_groups = model_data.groupby(['num_replicas', 'tensor_parallel_size'])
                
                strategies = []
                vidur_costs = []
                vllm_costs = []
                
                for (replicas, tp), group in config_groups:
                    min_total_cost = group['total_cost'].min()
                    strategies.append(f"R{replicas}_TP{tp}")
                    vidur_costs.append(min_total_cost)
                    
                    # Find corresponding vLLM cost
                    vllm_cost = None
                    search_key = (model_name, qps_value, replicas, tp)
                    print(f"Debug: Searching for vLLM match: {search_key}")
                    
                    for (vllm_model, vllm_qps, vllm_replicas, vllm_tp), cost in vllm_data.items():
                        if (model_name == vllm_model and 
                            abs(qps_value - vllm_qps) < 0.1 and 
                            replicas == vllm_replicas and 
                            tp == vllm_tp):
                            vllm_cost = cost
                            print(f"Debug: Found vLLM match: {(vllm_model, vllm_qps, vllm_replicas, vllm_tp)} -> ${cost:.4f}")
                            break
                    
                    if vllm_cost is None:
                        print(f"Debug: No vLLM match found for {search_key}")
                    
                    vllm_costs.append(vllm_cost)
                
                if not strategies:
                    ax.text(0.5, 0.5, f"No strategies for {model_name}", ha='center', va='center')
                    ax.set_title(f'{model_name} - {instance_type} - QPS {qps_value}')
                    continue
                
                # Create bar plot with both Vidur and vLLM costs
                x = np.arange(len(strategies))
                width = 0.35
                
                bars1 = ax.bar(x - width/2, vidur_costs, width, alpha=0.7, color='lightcoral', 
                              edgecolor='black', label='Vidur Simulator')
                
                # Only plot vLLM bars where data exists
                vllm_costs_filtered = [cost if cost is not None else 0 for cost in vllm_costs]
                vllm_mask = [cost is not None for cost in vllm_costs]
                
                if any(vllm_mask):
                    bars2 = ax.bar(x + width/2, vllm_costs_filtered, width, alpha=0.7, color='lightblue', 
                                  edgecolor='black', label='vLLM Benchmark')
                
                # Find min total cost for percentage calculation
                min_vidur_cost = min(vidur_costs)
                
                # Add percentage text boxes for Vidur
                for j, (strategy, value) in enumerate(zip(strategies, vidur_costs)):
                    if value == min_vidur_cost:
                        percentage_text = '0%'
                    else:
                        percentage_diff = ((value - min_vidur_cost) / min_vidur_cost) * 100
                        percentage_text = f'+{percentage_diff:.1f}%'
                    
                    ax.text(j - width/2, value + max(vidur_costs) * 0.02, percentage_text, 
                           ha='center', va='bottom', fontweight='bold', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.7))
                
                # Add percentage text boxes for vLLM where data exists
                if any(vllm_mask):
                    valid_vllm_costs = [cost for cost in vllm_costs if cost is not None]
                    if valid_vllm_costs:
                        min_vllm_cost = min(valid_vllm_costs)
                        for j, (strategy, value) in enumerate(zip(strategies, vllm_costs)):
                            if value is not None:
                                if value == min_vllm_cost:
                                    percentage_text = '0%'
                                else:
                                    percentage_diff = ((value - min_vllm_cost) / min_vllm_cost) * 100
                                    percentage_text = f'+{percentage_diff:.1f}%'
                                
                                ax.text(j + width/2, value + max(vidur_costs) * 0.02, percentage_text, 
                                       ha='center', va='bottom', fontweight='bold', fontsize=9,
                                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightcyan', alpha=0.7))
                
                # Formatting
                ax.set_title(f'{model_name} - QPS {qps_value}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Strategy (Replicas_TensorParallel)', fontsize=12)
                ax.set_ylabel('Min Total Cost ($)', fontsize=12)
                ax.set_xticks(x)
                ax.set_xticklabels(strategies, rotation=45)
                ax.grid(True, alpha=0.3, axis='y')
                ax.legend()
                
                # Set y-limit to accommodate both datasets
                max_cost = max(max(vidur_costs), max([c for c in vllm_costs if c is not None], default=0))
                ax.set_ylim(0, max_cost * 1.2)
            
            # Hide unused subplots
            for i in range(n_models, len(axes)):
                axes[i].set_visible(False)
            
            # Add overall title
            fig.suptitle(f'{instance_type} - QPS {qps_value} (Single Node) - Vidur vs vLLM Benchmark', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save the plot in parallel_figs directory
            filename = f"parallelism_strategies_{instance_type}_qps{qps_value}.png"
            filepath = os.path.join(parallel_figs_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved parallelism strategies plot: {filepath}")
    
    # Create parallelism strategies plots for all network devices
    network_devices = df['network_device'].unique()
    for device in network_devices:
        plot_parallelism_strategies(instance_type=device)
    
    print("Analysis complete.")
    print(f"Raw data saved: {os.path.join(output_dir, 'config_optimizer_results.csv')}")
    print(f"Plots saved:")
    print(f"  - {os.path.join(output_dir, 'qps_scatter.png')} - QPS performance metrics")
    print(f"  - {os.path.join(output_dir, 'qps_per_dollar_scatter.png')} - Cost efficiency metrics")
    print(f"  - {os.path.join(output_dir, 'parallel_figs/')} - Parallelism strategies plots with vLLM benchmark comparison")
