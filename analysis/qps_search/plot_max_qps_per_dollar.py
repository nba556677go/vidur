import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import csv

class ConfigOptimizerPlotter:
    """
    A class to plot configuration optimizer results, focusing on max QPS per dollar
    for each parallelism/replica configuration under different network devices.
    """
    
    def _get_instance_order(self, instances):
        """Return instances in fixed order: g5, g6, g6e, p4d, p5"""
        order_map = {'a10g_g5': 0, 'l4_g6': 1, 'l40s_g6e48': 2, 'p4d_a100_40g_nvlink': 3, 'h100_p5': 4}
        return sorted(instances, key=lambda x: order_map.get(x, 999))
    
    def __init__(self, csv_file, input_param=None, output_dir="max_qps", slo_limit=0.25, exec_slo=7.8, inter_token_slo=0.015):
        """
        Initialize the plotter with CSV data and input parameters.
        
        Args:
            csv_file (str): Path to the CSV file containing config optimizer results
            input_param (dict): Dictionary with filtering parameters like 
                               {'prefill_tokens': 300, 'decode_tokens': 3}
            output_dir (str): Directory to save the plots (default: "max_qps")
            slo_limit (float): SLO limit for TTFT in seconds (default: 0.25)
            exec_slo (float): SLO limit for total execution time in seconds (default: 7.8)
            inter_token_slo (float): SLO limit for inter-token latency in seconds (default: 0.015)
        """
        self.csv_file = csv_file
        self.input_param = input_param or {'prefill_tokens': 300, 'decode_tokens': 3}
        self.output_dir = output_dir
        self.df = None
        self.filtered_df = None
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # SLO limits
        self.slo_limit = slo_limit  # 250ms for TTFT
        self.exec_slo = exec_slo    # slo for total execution time
        self.inter_token_slo = inter_token_slo  # 15ms in seconds
        
    def load_and_filter_data(self):
        """Load CSV data and apply input parameter filters."""
        print(f"Loading data from {self.csv_file}")
        self.df = pd.read_csv(self.csv_file)
        
        print(f"Original data shape: {self.df.shape}")
        
        # Apply input parameter filters
        filter_conditions = []
        for param, value in self.input_param.items():
            if param in self.df.columns:
                filter_conditions.append(self.df[param] == value)
                print(f"Filtering by {param} = {value}")
            else:
                print(f"Warning: Column {param} not found in data")
        
        if filter_conditions:
            combined_filter = filter_conditions[0]
            for condition in filter_conditions[1:]:
                combined_filter = combined_filter & condition
            self.filtered_df = self.df[combined_filter].copy()
        else:
            self.filtered_df = self.df.copy()
            
        print(f"Filtered data shape: {self.filtered_df.shape}")
        
        if len(self.filtered_df) == 0:
            print("Warning: No data remains after filtering!")
            return
            
        # Print unique models found
        unique_models = self.filtered_df['model_name'].unique()
        print(f"Unique models found: {unique_models}")
        
    def get_max_qps_per_dollar_data(self, model_df):
        """
        For each unique combination of parallelism parameters and network_device,
        find the configuration with maximum QPS per dollar that meets SLO requirements.
        
        Args:
            model_df (DataFrame): Data for a specific model
            
        Returns:
            DataFrame: Data with max QPS per dollar for each configuration group (SLO compliant only)
        """
        # Group by parallelism parameters and network device
        groupby_cols = ['network_device', 'tensor_parallel_size', 'num_pipeline_stages', 'num_replicas']
        
        # Find the row with max qps_per_dollar for each group (SLO compliant only)
        max_qps_per_dollar_data = []
        
        for group_keys, group_df in model_df.groupby(groupby_cols):
            if len(group_df) > 0:
                # Filter for SLO compliant configurations only
                slo_compliant = group_df[(group_df['p99_ttft'] <= self.slo_limit) & 
                                        (group_df['p99_exec_time'] <= self.exec_slo)]
                
                if len(slo_compliant) > 0:
                    # Find the row with maximum qps_per_dollar in SLO compliant group
                    max_idx = slo_compliant['qps_per_dollar'].idxmax()
                    max_row = slo_compliant.loc[max_idx].copy()
                    max_qps_per_dollar_data.append(max_row)
        
        result_df = pd.DataFrame(max_qps_per_dollar_data)
        return result_df
    
    def create_subplots_for_model(self, model_name, model_data, plot_type='qps'):
        """
        Create the 4-subplot figure for a specific model.
        
        Args:
            model_name (str): Name of the model
            model_data (DataFrame): Filtered data for this model
            plot_type (str): Either 'qps' or 'qps_per_dollar'
        """
        # Get max QPS per dollar data
        max_data = self.get_max_qps_per_dollar_data(model_data)
        
        if len(max_data) == 0:
            print(f"No data found for model {model_name}")
            return
            
        print(f"Plotting {len(max_data)} max QPS per dollar points for model {model_name}")
        
        # Create color mapping for network devices
        unique_network_devices = max_data['network_device'].unique()
        colors = plt.cm.viridis(np.linspace(0, 1, max(len(unique_network_devices), 1)))
        color_map = {device: colors[i] for i, device in enumerate(unique_network_devices)}
        
        # Determine y-axis values and titles based on plot type
        if plot_type == 'qps_per_dollar':
            y_col = 'qps_per_dollar'
            y_label = 'QPS per Dollar'
            fig_title_prefix = 'Cost Efficiency'
            color_metric = 'qps_per_dollar'
            color_label = 'QPS per Dollar'
        else:
            y_col = 'qps'
            y_label = 'QPS'
            fig_title_prefix = 'Performance'
            color_metric = 'qps'
            color_label = 'QPS'
        
        # Find best configs
        slo_compliant = max_data[(max_data['p99_ttft'] <= self.slo_limit) & 
                                (max_data['p99_exec_time'] <= self.exec_slo)]
        
        if len(slo_compliant) > 0:
            if plot_type == 'qps_per_dollar':
                best_config = slo_compliant.loc[slo_compliant['qps_per_dollar'].idxmax()]
            else:   
                best_config = slo_compliant.loc[slo_compliant['qps'].idxmax()]
        else:
            print(f"Warning: No SLO compliant configs for {model_name}, using best TTFT")
            best_config = max_data.loc[max_data['p99_ttft'].idxmin()]
        
        # Create best config description
        best_desc = (f"Best {y_label} Config for {model_name}: "
                    f"PP={best_config['num_pipeline_stages']}, "
                    f"TP={best_config['tensor_parallel_size']}, "
                    f"Replicas={best_config['num_replicas']}, "
                    f"Nodes={best_config['nodes_needed']}, "
                    f"Scheduler={best_config['scheduler_type']}, "
                    f"SKU={best_config['network_device']}, "
                    f"QPS={best_config['qps']:.2f}, "
                    f"QPS/$={best_config['qps_per_dollar']:.4f}")
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(18, 16), 
                                gridspec_kw={'width_ratios': [1, 1.2], 'height_ratios': [1, 1]})
        fig.text(0.5, 0.97, best_desc, ha='center', fontsize=17)
        
        # Check data availability
        has_exec_time = all(pd.notna(max_data['p99_exec_time']))
        has_inter_token_latency = pd.notna(max_data['p99_inter_token_latency']).any()
        
        # Subplot 1: Y vs P99 TTFT
        ax = axes[0, 0]
        ax.set_title(f"{y_label} vs P99 Time to First Token", fontsize=19)
        
        for device in unique_network_devices:
            subset = max_data[max_data['network_device'] == device]
            ax.scatter(subset['p99_ttft'], subset[y_col], 
                      label=device, color=color_map[device], s=80, alpha=0.7)
        
        ax.axvline(x=self.slo_limit, color='red', linestyle='--', 
                  label=f'SLO Limit ({self.slo_limit*1000}ms)')
        ax.axvspan(0, self.slo_limit, alpha=0.1, color='green', label='SLO Compliant Region')
        
        ax.set_xlabel("Time to First Token - P99 (s)", fontsize=17)
        ax.set_ylabel(y_label, fontsize=17)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Network Device", fontsize=13)
        
        # Subplot 2: Y vs P99 Request Total Latency
        ax = axes[0, 1]
        ax.set_title(f"{y_label} vs P99 Request Total Latency", fontsize=19)
        
        if has_exec_time:
            for device in unique_network_devices:
                subset = max_data[max_data['network_device'] == device]
                ax.scatter(subset['p99_exec_time'], subset[y_col], 
                          label=device, color=color_map[device], s=80, alpha=0.7)
            
            ax.axvline(x=self.exec_slo, color='red', linestyle='--', 
                      label=f'SLO Limit ({self.exec_slo}s)')
            ax.axvspan(0, self.exec_slo, alpha=0.1, color='green', label='SLO Compliant Region')
            
            ax.set_xlabel("Request Execution Time - P99 (s)", fontsize=17)
            ax.set_ylabel(y_label, fontsize=17)
            ax.grid(True, alpha=0.3)
            ax.legend(title="Network Device", fontsize=13)
        else:
            ax.text(0.5, 0.5, "P99 Request Total Latency metrics not available", 
                   ha='center', va='center', fontsize=19)
        
        # Subplot 3: Y vs P99 Inter-Token Latency
        ax = axes[1, 0]
        ax.set_title(f"{y_label} vs P99 Inter-Token Latency", fontsize=19)
        
        if has_inter_token_latency:
            for device in unique_network_devices:
                subset = max_data[max_data['network_device'] == device]
                ax.scatter(subset['p99_inter_token_latency'], subset[y_col], 
                          label=device, color=color_map[device], s=80, alpha=0.7)
            
            ax.axvline(x=self.inter_token_slo, color='red', linestyle='--', 
                      label=f'Inter-Token SLO ({self.inter_token_slo*1000}ms)')
            ax.axvspan(0, self.inter_token_slo, alpha=0.1, color='green', 
                      label='SLO Compliant Region')
            
            ax.set_xlabel("Inter-Token Latency - P99 (s)", fontsize=17)
            ax.set_ylabel(y_label, fontsize=17)
            ax.grid(True, alpha=0.3)
            ax.legend(title="Network Device", fontsize=13)
        else:
            ax.text(0.5, 0.5, "P99 Inter-Token Latency metrics not available", 
                   ha='center', va='center', fontsize=19)
        
        # Subplot 4: P99 Exec Time vs P99 TTFT, colored by metric
        ax = axes[1, 1]
        ax.set_title(f"P99 Exec Time vs P99 TTFT (Colored by {color_label})", fontsize=19)
        
        if has_exec_time:
            scatter = ax.scatter(max_data['p99_ttft'], max_data['p99_exec_time'], 
                               c=max_data[color_metric], cmap='viridis', s=100, alpha=0.7)
            
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = plt.colorbar(scatter, cax=cax)
            cbar.set_label(color_label, fontsize=17)
            
            # Highlight best config
            ax.scatter(best_config['p99_ttft'], best_config['p99_exec_time'], 
                      color='gold', s=250, marker='*', 
                      label=f"Best: {best_config['network_device']} TP{best_config['tensor_parallel_size']}/PP{best_config['num_pipeline_stages']}", 
                      edgecolor='black', zorder=10)
            
            # Add SLO limit lines
            ax.axvline(x=self.slo_limit, color='red', linestyle='--', 
                      label=f'TTFT SLO ({self.slo_limit*1000}ms)')
            ax.axhline(y=self.exec_slo, color='red', linestyle=':', 
                      label=f'Exec SLO ({self.exec_slo}s)')
            
            # Add SLO compliant region
            ax.fill_between([0, self.slo_limit], 0, self.exec_slo, 
                           alpha=0.1, color='green', label='SLO Compliant Region')
            
            ax.set_xlabel("Time to First Token - P99 (s)", fontsize=17)
            ax.set_ylabel("Execution Time - P99 (s)", fontsize=17)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=13)
        else:
            ax.text(0.5, 0.5, "P99 Execution Time metrics not available", 
                   ha='center', va='center', fontsize=19)
        
        # Save figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Clean model name for filename
        clean_model_name = model_name.replace('/', '_').replace(' ', '_')
        filename = f"max_qps_per_dollar_{plot_type}_{clean_model_name}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save CSV data for this figure
        csv_filename = f"max_qps_per_dollar_{plot_type}_{clean_model_name}_data.csv"
        csv_filepath = os.path.join(self.output_dir, csv_filename)
        max_data.to_csv(csv_filepath, index=False)
        
        # Save SLO compliant data
        slo_compliant_data = max_data[(max_data['p99_ttft'] <= self.slo_limit) & 
                                     (max_data['p99_exec_time'] <= self.exec_slo)]
        if len(slo_compliant_data) > 0:
            pass
            slo_csv_filename = f"slo_compliant_{plot_type}_{clean_model_name}_data.csv"
            slo_csv_filepath = os.path.join(self.output_dir, slo_csv_filename)
            slo_compliant_data.to_csv(slo_csv_filepath, index=False)
            print(f"Saved SLO compliant data: {slo_csv_filepath}")
        
        print(f"Saved plot: {filepath}")
        print(f"Saved data: {csv_filepath}")
    
    def plot_max_qps_per_dollar_barchart(self):
        """Create bar chart showing max QPS per dollar for each instance type, with subplots for each model (multinode version)."""
        if self.filtered_df is None:
            print("No data loaded. Call load_and_filter_data() first.")
            return
            
        unique_models = self.filtered_df['model_name'].unique()
        n_models = len(unique_models)
        
        if n_models == 0:
            print("No models found in data.")
            return
        
        # Create subplots - square layout for 2*n_models subplots
        total_subplots = n_models * 2
        n_cols = int(np.ceil(np.sqrt(total_subplots)))
        n_rows = int(np.ceil(total_subplots / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
        axes = axes.flatten() if total_subplots > 1 else [axes]
        
        # Plot each model (2 subplots per model)
        for i, model_name in enumerate(unique_models):
            qps_ax = axes[i*2]     # QPS per dollar subplot
            cost_ax = axes[i*2+1]  # Total cost subplot
            
            model_data = self.filtered_df[self.filtered_df['model_name'] == model_name]
            
            # Get max QPS per dollar data (SLO compliant only)
            max_qps_data = self.get_max_qps_per_dollar_data(model_data)
            # Get min total cost data (SLO compliant only)
            min_cost_data = self.get_min_cost_data(model_data)
            
            if len(max_qps_data) == 0 or len(min_cost_data) == 0:
                qps_ax.text(0.5, 0.5, f"No SLO compliant data for {model_name}", ha='center', va='center')
                qps_ax.set_title(f'{model_name} - QPS per Dollar (Multinode)')
                cost_ax.text(0.5, 0.5, f"No SLO compliant data for {model_name}", ha='center', va='center')
                cost_ax.set_title(f'{model_name} - Total Cost (Multinode)')
                continue
            
            # Prepare QPS per dollar data - get max for each instance
            qps_instances = self._get_instance_order(max_qps_data['network_device'].unique())
            qps_values = []
            qps_nodes = []
            qps_qps_values = []
            
            for inst in qps_instances:
                inst_data = max_qps_data[max_qps_data['network_device'] == inst]
                if len(inst_data) > 0:
                    # Get the config with max QPS per dollar for this instance
                    best_config = inst_data.loc[inst_data['qps_per_dollar'].idxmax()]
                    qps_values.append(best_config['qps_per_dollar'])
                    qps_nodes.append(best_config['nodes_needed'])
                    qps_qps_values.append(best_config['qps'])
                else:
                    qps_values.append(0)
                    qps_nodes.append(0)
                    qps_qps_values.append(0)
            
            # Prepare min cost data - get min for each instance
            cost_instances = self._get_instance_order(min_cost_data['network_device'].unique())
            cost_values = []
            cost_nodes = []
            cost_qps_values = []
            
            for inst in cost_instances:
                inst_data = min_cost_data[min_cost_data['network_device'] == inst]
                if len(inst_data) > 0:
                    # Get the config with min total cost for this instance
                    best_config = inst_data.loc[inst_data['total_cost'].idxmin()]
                    cost_values.append(best_config['total_cost'])
                    cost_nodes.append(best_config['nodes_needed'])
                    cost_qps_values.append(best_config['qps'])
                else:
                    cost_values.append(0)
                    cost_nodes.append(0)
                    cost_qps_values.append(0)
            
            # Plot QPS per dollar (Vidur only - no vLLM for multinode)
            bars1 = qps_ax.bar(qps_instances, qps_values, alpha=0.7, color='lightcoral', edgecolor='black')
            
            # Find max QPS per dollar for this model
            model_max_qps_per_dollar = max([v for v in qps_values if v > 0]) if any(v > 0 for v in qps_values) else 0
            
            # Add percentage text boxes on QPS per dollar bars
            for j, (instance, value, nodes, qps) in enumerate(zip(qps_instances, qps_values, qps_nodes, qps_qps_values)):
                if value == model_max_qps_per_dollar and value > 0:
                    percentage_text = '0%'
                elif value > 0 and model_max_qps_per_dollar > 0:
                    percentage_diff = ((value - model_max_qps_per_dollar) / model_max_qps_per_dollar) * 100
                    percentage_text = f'{percentage_diff:.1f}%'
                else:
                    percentage_text = 'N/A'
                
                text_content = f"{percentage_text}\nQPS: {qps:.1f}\nNodes: {int(nodes)}"
                if qps_values and max(qps_values) > 0:
                    qps_ax.text(j, value + max(qps_values) * 0.02, text_content, 
                               ha='center', va='bottom', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            # Format QPS per dollar subplot
            qps_ax.set_title(f'{model_name} - Max QPS per Dollar (Multinode)', fontsize=14, fontweight='bold')
            qps_ax.set_ylabel('Max QPS per Dollar', fontsize=12)
            qps_ax.tick_params(axis='x', rotation=45)
            qps_ax.grid(True, alpha=0.3, axis='y')
            if qps_values and max(qps_values) > 0:
                qps_ax.set_ylim(0, max(qps_values) * 1.15)
            
            # Plot total cost (Vidur only - no vLLM for multinode)
            bars2 = cost_ax.bar(cost_instances, cost_values, alpha=0.7, color='lightcoral', edgecolor='black')
            
            # Find min cost for this model (best is lowest cost)
            model_min_cost = min([c for c in cost_values if c > 0]) if any(c > 0 for c in cost_values) else 0
            
            # Add percentage text boxes on total cost bars
            for j, (instance, cost, nodes, qps) in enumerate(zip(cost_instances, cost_values, cost_nodes, cost_qps_values)):
                if cost == model_min_cost and cost > 0:
                    percentage_text = '0%'
                elif cost > 0 and model_min_cost > 0:
                    percentage_diff = ((cost - model_min_cost) / model_min_cost) * 100
                    percentage_text = f'+{percentage_diff:.1f}%'
                else:
                    percentage_text = 'N/A'
                
                text_content = f"{percentage_text}\nQPS: {qps:.1f}\nNodes: {int(nodes)}"
                if cost_values and max(cost_values) > 0:
                    cost_ax.text(j, cost + max(cost_values) * 0.02, text_content, 
                               ha='center', va='bottom', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
            
            # Format total cost subplot
            cost_ax.set_title(f'{model_name} - Min Total Cost (Multinode)', fontsize=14, fontweight='bold')
            cost_ax.set_xlabel('Instance Type', fontsize=12)
            cost_ax.set_ylabel('Min Total Cost ($)', fontsize=12)
            cost_ax.tick_params(axis='x', rotation=45)
            cost_ax.grid(True, alpha=0.3, axis='y')
            if cost_values and max(cost_values) > 0:
                cost_ax.set_ylim(0, max(cost_values) * 1.15)
        
        # Hide unused subplots in the square grid
        for i in range(total_subplots, len(axes)):
            axes[i].set_visible(False)
        
        # Add overall title
        fig.suptitle('Max QPS per Dollar & Min Total Cost by Instance Type (Multinode)\n(Percentages show difference from model optimum, SLO Compliant Only)', 
                    fontsize=16, fontweight='bold', y=0.99)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the plot
        filename = "max_qps_per_dollar_barchart_multinode.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved multinode bar chart: {filepath}")
        
        # Create single node version
        self._plot_single_node_barchart()
    
    def get_device_costs(self):
        """Get device costs from EC2 pricing CSV."""
        device_costs = {
            'p4d_a100_40g_nvlink': {'cost': None, 'gpus_per_node': 8},
            'h100_p5': {'cost': None, 'gpus_per_node': 8},
            'l40s_g6e48': {'cost': None, 'gpus_per_node': 8},
            'a10g_g5': {'cost': None, 'gpus_per_node': 8},
            'l4_g6': {'cost': None, 'gpus_per_node': 8},
        }
        
        instance_to_device = {
            'p4d.24xlarge': 'p4d_a100_40g_nvlink',
            'p5.48xlarge': 'h100_p5',
            'g6e.48xlarge': 'l40s_g6e48',
            'g5.48xlarge': 'a10g_g5',
            'g6.48xlarge': 'l4_g6'
        }
        
        csv_path = os.path.join(os.path.dirname(__file__), "Plannable_Public_EC2_US_East.csv")
        
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                f.readline()  # Skip first line
                reader = csv.DictReader(f, delimiter=';')
                
                for row in reader:
                    instance_type = row.get('Instance Type', '').strip()
                    if instance_type in instance_to_device:
                        device = instance_to_device[instance_type]
                        try:
                            # Use only 2025 IMR pricing
                            imr_cost_str = row.get('2025 IMR', '0').strip()
                            
                            if imr_cost_str and float(imr_cost_str) > 0:
                                cost = float(imr_cost_str)
                                device_costs[device]['cost'] = cost
                        except (ValueError, TypeError):
                            pass
        
        # Set fallback costs if not found (using 2025 IMR values)
        fallback_costs = {
            'p4d_a100_40g_nvlink': 4.73,  # 2025 IMR
            'h100_p5': 11.85,             # 2025 IMR
            'l40s_g6e48': 4.68,           # 2025 IMR
            'a10g_g5': 2.44,              # 2025 IMR
            'l4_g6': 2.20                 # 2025 IMR
        }
        
        for device, info in device_costs.items():
            if info['cost'] is None:
                info['cost'] = fallback_costs.get(device, 1.0)
        
        return device_costs
    
    def load_vllm_benchmark_data(self, instance_type):
        """Load vLLM benchmark data for comparison."""
        vllm_data = {}
        device_costs = self.get_device_costs()
        
        vllm_path = "/home/ec2-user/vidur-simulator/analysis/vllm_bench_results/qu_brand/vidur_qps"
        
        if os.path.exists(vllm_path):
            csv_files = glob.glob(os.path.join(vllm_path, "latency_metrics_*.csv"))
            print(f"Debug: Found {len(csv_files)} vLLM CSV files")
            
            for csv_file in csv_files:
                try:
                    vllm_df = pd.read_csv(csv_file)
                    print(f"Debug: Loaded {len(vllm_df)} rows from {csv_file}")
                    
                    # Filter for the specific instance type
                    device_filter_map = {
                        'p4d_a100_40g_nvlink': 'p4d_a100_40g_nvlink',
                        'a10g_g5': 'a10g_g5',
                        'h100_p5': 'h100_p5',
                        'l4_g6': 'l4_g6',
                        'l40s_g6e48': 'l40s_g6e48'
                    }
                    
                    device_name = device_filter_map.get(instance_type)
                    if device_name:
                        vllm_df_filtered = vllm_df[vllm_df['Device'] == device_name]
                        print(f"Debug: After filtering for {device_name}: {len(vllm_df_filtered)} rows")
                    else:
                        vllm_df_filtered = vllm_df
                        print(f"Debug: No device filter applied, using all {len(vllm_df_filtered)} rows")
                    
                    for _, row in vllm_df_filtered.iterrows():
                        model = row['Model']
                        qps = row['QPS']
                        tp = row['Tensor_Parallel']
                        pp = row.get('Pipeline_Parallel', 1)
                        replicas = row['Num_Replicas']
                        process_time = row['Process_Time_Seconds']
                        
                        # Calculate cost using process time
                        device_cost_per_hour = device_costs.get(instance_type, {}).get('cost', 0)
                        gpus_per_node = device_costs.get(instance_type, {}).get('gpus_per_node', 8)
                        
                        replica_per_node = gpus_per_node / (tp * pp)
                        nodes_needed = np.ceil(replicas / replica_per_node)
                        
                        total_cost_per_hour = device_cost_per_hour * nodes_needed
                        total_cost = total_cost_per_hour * (process_time / 3600.0)
                        
                        key = (model, qps, replicas, tp)
                        if key not in vllm_data or vllm_data[key] > total_cost:
                            vllm_data[key] = total_cost
                            
                except Exception as e:
                    print(f"Error loading vLLM data from {csv_file}: {e}")
                    continue
        
        print(f"Debug: Loaded {len(vllm_data)} vLLM entries for {instance_type}")
        if len(vllm_data) > 0:
            print(f"Debug: Sample vLLM entries: {list(vllm_data.items())[:3]}")
        else:
            print(f"Debug: No vLLM data found for {instance_type}. Available devices in CSV:")
            if csv_files:
                sample_df = pd.read_csv(csv_files[0])
                if 'Device' in sample_df.columns:
                    print(f"Debug: Available devices: {sample_df['Device'].unique()}")
        
        return vllm_data
    
    def plot_parallelism_strategies(self, instance_type='l40s_g6e48'):
        """Plot parallelism strategies for a specific instance type with nodes_needed==1."""
        if self.filtered_df is None:
            print("No data loaded. Call load_and_filter_data() first.")
            return
        
        # Create parallel_figs directory
        parallel_figs_dir = os.path.join(self.output_dir, 'parallel_figs')
        os.makedirs(parallel_figs_dir, exist_ok=True)
        
        # Load vLLM benchmark data
        vllm_data = self.load_vllm_benchmark_data(instance_type)
        
        # Filter for specific instance and nodes_needed==1
        instance_data = self.filtered_df[
            (self.filtered_df['network_device'] == instance_type) & 
            (self.filtered_df['nodes_needed'] == 1)
        ].copy()
        
        if len(instance_data) == 0:
            print(f"No data found for {instance_type} with nodes_needed==1")
            return
        
        unique_models = instance_data['model_name'].unique()
        n_models = len(unique_models)
        
        if n_models == 0:
            print("No models found in filtered data.")
            return
        
        # Create subplots for each model
        if n_models == 1:
            fig, axes = plt.subplots(1, 1, figsize=(12, 8))
            axes = [axes]
        else:
            n_cols = int(np.ceil(np.sqrt(n_models)))
            n_rows = int(np.ceil(n_models / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
            axes = axes.flatten() if n_models > 1 else [axes]
        
        for i, model_name in enumerate(unique_models):
            ax = axes[i]
            model_data = instance_data[instance_data['model_name'] == model_name]
            
            # Apply SLO constraints and get min total cost per config
            print(f"Debug: {model_name} on {instance_type} - Total configs: {len(model_data)}")
            print(f"Debug: TTFT SLO limit: {self.slo_limit}s, Exec SLO limit: {self.exec_slo}s")
            
            slo_compliant = model_data[
                (model_data['p99_ttft'] <= self.slo_limit) & 
                (model_data['p99_exec_time'] <= self.exec_slo)
            ]
            
            print(f"Debug: SLO compliant configs: {len(slo_compliant)}")
            if len(model_data) > 0:
                print(f"Debug: TTFT range: {model_data['p99_ttft'].min():.3f} - {model_data['p99_ttft'].max():.3f}")
                print(f"Debug: Exec time range: {model_data['p99_exec_time'].min():.3f} - {model_data['p99_exec_time'].max():.3f}")
            
            if len(slo_compliant) == 0:
                ax.text(0.5, 0.5, f"No SLO compliant configs for {model_name}", ha='center', va='center')
                ax.set_title(f'{model_name} - {instance_type}')
                continue
            
            # Group by parallelism strategy and get min total cost
            config_groups = slo_compliant.groupby(['num_replicas', 'tensor_parallel_size'])
            print(f"Debug: Available parallelism strategies: {list(config_groups.groups.keys())}")
            
            # Check for missing (2,2) config
            r2_tp2_configs = model_data[(model_data['num_replicas'] == 2) & (model_data['tensor_parallel_size'] == 2)]
            print(f"Debug: R2_TP2 total configs: {len(r2_tp2_configs)}")
            if len(r2_tp2_configs) > 0:
                r2_tp2_slo = r2_tp2_configs[(r2_tp2_configs['p99_ttft'] <= self.slo_limit) & (r2_tp2_configs['p99_exec_time'] <= self.exec_slo)]
                print(f"Debug: R2_TP2 SLO compliant: {len(r2_tp2_slo)}")
                print(f"Debug: R2_TP2 TTFT range: {r2_tp2_configs['p99_ttft'].min():.3f} - {r2_tp2_configs['p99_ttft'].max():.3f}")
                print(f"Debug: R2_TP2 exec range: {r2_tp2_configs['p99_exec_time'].min():.3f} - {r2_tp2_configs['p99_exec_time'].max():.3f}")
            
            strategies = []
            vidur_costs = []
            vllm_costs = []
            
            for (replicas, tp), group in config_groups:
                min_cost_config = group.loc[group['total_cost'].idxmin()]
                min_total_cost = min_cost_config['total_cost']
                selected_qps = min_cost_config['qps']
                
                strategies.append(f"R{replicas}_TP{tp}")
                vidur_costs.append(min_total_cost)
                
                # Find corresponding vLLM cost using selected config's QPS and parallelism
                vllm_cost = None
                search_key = (model_name, selected_qps, replicas, tp)
                print(f"Debug: Searching for vLLM match: {search_key}")
                
                for (vllm_model, vllm_qps, vllm_replicas, vllm_tp), cost in vllm_data.items():
                    if (model_name == vllm_model and 
                        abs(selected_qps - vllm_qps) < 0.1 and 
                        replicas == vllm_replicas and 
                        tp == vllm_tp):
                        vllm_cost = cost
                        print(f"Debug: Found vLLM match: {(vllm_model, vllm_qps, vllm_replicas, vllm_tp)} -> ${cost:.4f}")
                        break
                
                if vllm_cost is None:
                    print(f"Debug: No vLLM match found for {search_key}")
                
                vllm_costs.append(vllm_cost)
            
            if not strategies:
                ax.text(0.5, 0.5, f"No valid strategies for {model_name}", ha='center', va='center')
                ax.set_title(f'{model_name} - {instance_type}')
                continue
            
            # Create bar plot with both Vidur and vLLM costs
            x = np.arange(len(strategies))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, vidur_costs, width, alpha=0.7, color='lightcoral', 
                          edgecolor='black', label='Vidur Predicted')
            
            # Only plot vLLM bars where data exists
            vllm_costs_filtered = [cost if cost is not None else 0 for cost in vllm_costs]
            vllm_mask = [cost is not None for cost in vllm_costs]
            
            if any(vllm_mask):
                bars2 = ax.bar(x + width/2, vllm_costs_filtered, width, alpha=0.7, color='skyblue',
                              edgecolor='black', label='vLLM Actual')
            
            # Find min total cost for percentage calculation
            min_total_cost = min(vidur_costs)
            
            # Add percentage text boxes with QPS info for Vidur costs
            for j, (strategy, value) in enumerate(zip(strategies, vidur_costs)):
                if value == min_total_cost:
                    percentage_text = '0%'
                else:
                    percentage_diff = ((value - min_total_cost) / min_total_cost) * 100
                    percentage_text = f'+{percentage_diff:.1f}%'
                
                # Get QPS for this strategy from the selected config
                replicas, tp = map(int, strategy.replace('R', '').replace('_TP', ' ').split())
                strategy_group = slo_compliant[(slo_compliant['num_replicas'] == replicas) & 
                                              (slo_compliant['tensor_parallel_size'] == tp)]
                selected_qps = strategy_group.loc[strategy_group['total_cost'].idxmin(), 'qps']
                
                text_content = f'{percentage_text}\nQPS: {selected_qps:.1f}'
                ax.text(j - width/2, value + max(vidur_costs) * 0.02, text_content, 
                       ha='center', va='bottom', fontweight='bold', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.7))
            
            # Add vLLM cost labels where available
            vllm_valid_costs = [cost for cost in vllm_costs if cost is not None]
            min_vllm_cost = min(vllm_valid_costs) if vllm_valid_costs else None
            
            for j, (vllm_cost, has_data) in enumerate(zip(vllm_costs, vllm_mask)):
                if has_data and vllm_cost is not None:
                    # Calculate percentage vs min vLLM cost
                    if min_vllm_cost and vllm_cost == min_vllm_cost:
                        percentage_text = '0%'
                    elif min_vllm_cost:
                        percentage_diff = ((vllm_cost - min_vllm_cost) / min_vllm_cost) * 100
                        percentage_text = f'+{percentage_diff:.1f}%'
                    else:
                        percentage_text = ''
                    
                    text_content = percentage_text
                    text_y = min(vllm_cost + max(vidur_costs) * 0.02, max(vidur_costs) * 1.1)
                    ax.text(j + width/2, text_y, text_content, 
                           ha='center', va='bottom', fontweight='bold', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.7))
                else:
                    # Show "No Data" for missing vLLM entries
                    ax.text(j + width/2, max(vidur_costs) * 0.02, 'No Data', 
                           ha='center', va='bottom', fontweight='bold', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', alpha=0.7))
            
            # Formatting
            ax.set_title(f'{model_name} - {instance_type}\nParallelism Strategies (Nodes=1)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Strategy (Replicas_TensorParallel)', fontsize=12)
            ax.set_ylabel('Min Total Cost ($)', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(strategies, rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(fontsize=10)
            ax.set_ylim(0, max(vidur_costs) * 1.15)
        
        # Hide unused subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        # Add overall title
        fig.suptitle(f'Parallelism Strategies for {instance_type} (Single Node)\nMin Total Cost under SLO Constraints', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the plot in parallel_figs directory
        filename = f"parallelism_strategies_{instance_type}.png"
        filepath = os.path.join(parallel_figs_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved parallelism strategies plot: {filepath}")
    
    def get_min_cost_data(self, model_df):
        """
        For each unique network_device, find the configuration with minimum total cost 
        that meets SLO requirements.
        
        Args:
            model_df (DataFrame): Data for a specific model
            
        Returns:
            DataFrame: Data with min total cost for each network device (SLO compliant only)
        """
        min_cost_data = []
        
        for device, device_df in model_df.groupby('network_device'):
            # Filter for SLO compliant configurations only
            slo_compliant = device_df[(device_df['p99_ttft'] <= self.slo_limit) & 
                                     (device_df['p99_exec_time'] <= self.exec_slo)]
            
            if len(slo_compliant) > 0:
                # Find the row with minimum total_cost in SLO compliant group
                min_idx = slo_compliant['total_cost'].idxmin()
                min_row = slo_compliant.loc[min_idx].copy()
                min_cost_data.append(min_row)
        
        result_df = pd.DataFrame(min_cost_data)
        return result_df
    
    def _plot_single_node_barchart(self):
        """Create bar chart for single node configurations only (nodes_needed==1)."""
        if self.filtered_df is None:
            return
            
        # Filter for single node configurations
        single_node_df = self.filtered_df[self.filtered_df['nodes_needed'] == 1].copy()
        
        if len(single_node_df) == 0:
            print("No single node data found.")
            return
        
        unique_models = single_node_df['model_name'].unique()
        n_models = len(unique_models)
        
        if n_models == 0:
            return
        
        # Create subplots - square layout for 2*n_models subplots
        total_subplots = n_models * 2
        n_cols = int(np.ceil(np.sqrt(total_subplots)))
        n_rows = int(np.ceil(total_subplots / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
        axes = axes.flatten() if total_subplots > 1 else [axes]
        
        # Plot each model (2 subplots per model)
        for i, model_name in enumerate(unique_models):
            qps_ax = axes[i*2]     # QPS per dollar subplot
            cost_ax = axes[i*2+1]  # Total cost subplot
            
            model_data = single_node_df[single_node_df['model_name'] == model_name]
            
            # Get max QPS per dollar data
            max_qps_data = self.get_max_qps_per_dollar_data(model_data)
            # Get min total cost data
            min_cost_data = self.get_min_cost_data(model_data)
            
            if len(max_qps_data) == 0 or len(min_cost_data) == 0:
                qps_ax.text(0.5, 0.5, f"No data for {model_name}", ha='center', va='center')
                qps_ax.set_title(f'{model_name} - QPS per Dollar (Single Node)')
                cost_ax.text(0.5, 0.5, f"No data for {model_name}", ha='center', va='center')
                cost_ax.set_title(f'{model_name} - Total Cost (Single Node)')
                continue
            
            # Prepare QPS per dollar data
            qps_instances = self._get_instance_order(max_qps_data['network_device'].unique())
            qps_values = []
            for inst in qps_instances:
                inst_data = max_qps_data[max_qps_data['network_device'] == inst]
                if len(inst_data) > 0:
                    qps_values.append(inst_data['qps_per_dollar'].iloc[0])
                else:
                    qps_values.append(0)
            
            # Prepare min cost data
            cost_instances = self._get_instance_order(min_cost_data['network_device'].unique())
            cost_values = []
            for inst in cost_instances:
                inst_data = min_cost_data[min_cost_data['network_device'] == inst]
                if len(inst_data) > 0:
                    cost_values.append(inst_data['total_cost'].iloc[0])
                else:
                    cost_values.append(0)
            
            # Get vLLM data for QPS per dollar comparison
            qps_vllm_values = []
            for instance in qps_instances:
                inst_data = max_qps_data[max_qps_data['network_device'] == instance]
                if len(inst_data) > 0:
                    config = inst_data.iloc[0]
                    vllm_data = self.load_vllm_benchmark_data(instance)
                    vllm_cost = None
                    for (vllm_model, vllm_qps, vllm_replicas, vllm_tp), cost in vllm_data.items():
                        if (model_name == vllm_model and 
                            abs(config['qps'] - vllm_qps) < 0.1 and 
                            config['num_replicas'] == vllm_replicas and 
                            config['tensor_parallel_size'] == vllm_tp):
                            vllm_cost = cost
                            break
                    qps_vllm_values.append(vllm_cost / config['qps'] if vllm_cost else None)
                else:
                    qps_vllm_values.append(None)
            
            # Plot QPS per dollar with vLLM comparison
            x = np.arange(len(qps_instances))
            width = 0.35
            
            bars1 = qps_ax.bar(x - width/2, qps_values, width, alpha=0.7, color='lightcoral', 
                              edgecolor='black', label='Vidur')
            
            vllm_qps_filtered = [1/v if v else 0 for v in qps_vllm_values]
            vllm_qps_mask = [v is not None for v in qps_vllm_values]
            
            if any(vllm_qps_mask):
                bars2 = qps_ax.bar(x + width/2, vllm_qps_filtered, width, alpha=0.7, color='skyblue',
                                  edgecolor='black', label='vLLM')
            
            # Find max QPS per dollar for this model
            model_max_qps_per_dollar = max(qps_values) if qps_values else 0
            
            # Add percentage text boxes on QPS per dollar bars
            for j, (instance, value) in enumerate(zip(qps_instances, qps_values)):
                if value == model_max_qps_per_dollar and value > 0:
                    percentage_text = '0%'
                elif value > 0:
                    percentage_diff = ((value - model_max_qps_per_dollar) / model_max_qps_per_dollar) * 100
                    percentage_text = f'{percentage_diff:.1f}%'
                else:
                    percentage_text = 'N/A'
                
                # Get QPS for this instance
                inst_data = max_qps_data[max_qps_data['network_device'] == instance]
                qps_text = f"QPS: {inst_data['qps'].iloc[0]:.1f}" if len(inst_data) > 0 else "QPS: N/A"
                
                if qps_values:
                    qps_ax.text(j - width/2, value + max(qps_values) * 0.02, f"{percentage_text}\n{qps_text}", 
                               ha='center', va='bottom', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            # Add vLLM percentage textboxes
            vllm_qps_valid = [v for v in vllm_qps_filtered if v > 0]
            model_max_vllm_qps = max(vllm_qps_valid) if vllm_qps_valid else 0
            y_limit = max(qps_values) * 1.15 if qps_values else 1
            
            for j, (vllm_val, has_data) in enumerate(zip(vllm_qps_filtered, vllm_qps_mask)):
                if has_data and vllm_val > 0:
                    if vllm_val == model_max_vllm_qps:
                        vllm_percentage = '0%'
                    else:
                        vllm_diff = ((vllm_val - model_max_vllm_qps) / model_max_vllm_qps) * 100
                        vllm_percentage = f'{vllm_diff:.1f}%'
                    
                    text_y = min(vllm_val + max(qps_values) * 0.02, y_limit * 0.95) if qps_values else vllm_val * 1.02
                    qps_ax.text(j + width/2, text_y, vllm_percentage, 
                               ha='center', va='bottom', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
            
            # Format QPS per dollar subplot
            qps_ax.set_title(f'{model_name} - Max QPS per Dollar (Single Node)', fontsize=14, fontweight='bold')
            qps_ax.set_ylabel('Max QPS per Dollar', fontsize=12)
            qps_ax.set_xticks(x)
            qps_ax.set_xticklabels(qps_instances, rotation=45)
            qps_ax.grid(True, alpha=0.3, axis='y')
            qps_ax.legend(fontsize=10)
            if qps_values:
                qps_ax.set_ylim(0, max(qps_values) * 1.15)
            
            # Get vLLM data for cost comparison
            cost_vllm_values = []
            for instance in cost_instances:
                inst_data = min_cost_data[min_cost_data['network_device'] == instance]
                if len(inst_data) > 0:
                    config = inst_data.iloc[0]
                    vllm_data = self.load_vllm_benchmark_data(instance)
                    vllm_cost = None
                    for (vllm_model, vllm_qps, vllm_replicas, vllm_tp), cost in vllm_data.items():
                        if (model_name == vllm_model and 
                            abs(config['qps'] - vllm_qps) < 0.1 and 
                            config['num_replicas'] == vllm_replicas and 
                            config['tensor_parallel_size'] == vllm_tp):
                            vllm_cost = cost
                            break
                    cost_vllm_values.append(vllm_cost)
                else:
                    cost_vllm_values.append(None)
            
            # Plot total cost with vLLM comparison
            x = np.arange(len(cost_instances))
            width = 0.35
            
            bars1 = cost_ax.bar(x - width/2, cost_values, width, alpha=0.7, color='lightcoral', 
                               edgecolor='black', label='Vidur')
            
            vllm_cost_filtered = [v if v else 0 for v in cost_vllm_values]
            vllm_cost_mask = [v is not None for v in cost_vllm_values]
            
            if any(vllm_cost_mask):
                bars2 = cost_ax.bar(x + width/2, vllm_cost_filtered, width, alpha=0.7, color='skyblue',
                                   edgecolor='black', label='vLLM')
            
            # Find min cost for this model (best is lowest cost)
            model_min_cost = min([c for c in cost_values if c > 0]) if any(c > 0 for c in cost_values) else 0
            
            # Add percentage text boxes on total cost bars
            for j, (instance, cost) in enumerate(zip(cost_instances, cost_values)):
                if cost == model_min_cost and cost > 0:
                    percentage_text = '0%'
                elif cost > 0 and model_min_cost > 0:
                    percentage_diff = ((cost - model_min_cost) / model_min_cost) * 100
                    percentage_text = f'+{percentage_diff:.1f}%'
                else:
                    percentage_text = 'N/A'
                
                # Get QPS for this instance
                inst_data = min_cost_data[min_cost_data['network_device'] == instance]
                qps_text = f"QPS: {inst_data['qps'].iloc[0]:.1f}" if len(inst_data) > 0 else "QPS: N/A"
                
                if cost_values:
                    cost_ax.text(j - width/2, cost + max(cost_values) * 0.02, f"{percentage_text}\n{qps_text}", 
                               ha='center', va='bottom', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
            
            # Add vLLM cost percentage textboxes
            vllm_cost_valid = [v for v in vllm_cost_filtered if v > 0]
            model_min_vllm_cost = min(vllm_cost_valid) if vllm_cost_valid else 0
            y_limit = max(cost_values) * 1.15 if cost_values else 1
            
            for j, (vllm_cost, has_data) in enumerate(zip(vllm_cost_filtered, vllm_cost_mask)):
                if has_data and vllm_cost > 0:
                    if vllm_cost == model_min_vllm_cost:
                        vllm_percentage = '0%'
                    else:
                        vllm_diff = ((vllm_cost - model_min_vllm_cost) / model_min_vllm_cost) * 100
                        vllm_percentage = f'+{vllm_diff:.1f}%'
                    
                    text_y = min(vllm_cost + max(cost_values) * 0.02, y_limit * 0.95) if cost_values else vllm_cost * 1.02
                    cost_ax.text(j + width/2, text_y, vllm_percentage, 
                               ha='center', va='bottom', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
            
            # Format total cost subplot
            cost_ax.set_title(f'{model_name} - Min Total Cost (Single Node)', fontsize=14, fontweight='bold')
            cost_ax.set_xlabel('Instance Type', fontsize=12)
            cost_ax.set_ylabel('Min Total Cost ($)', fontsize=12)
            cost_ax.set_xticks(x)
            cost_ax.set_xticklabels(cost_instances, rotation=45)
            cost_ax.grid(True, alpha=0.3, axis='y')
            cost_ax.legend(fontsize=10)
            if cost_values:
                cost_ax.set_ylim(0, max(cost_values) * 1.15)
        
        # Hide unused subplots in the square grid
        for i in range(total_subplots, len(axes)):
            axes[i].set_visible(False)
        
        # Add overall title
        fig.suptitle('Max QPS per Dollar & Total Cost by Instance Type (Single Node Only)\n(Percentages show difference from model max)', 
                    fontsize=16, fontweight='bold', y=0.99)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the plot
        filename = "max_qps_per_dollar_barchart_single_node.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved single node bar chart: {filepath}")
        #save single_nodedf to csv
        single_node_df.to_csv(os.path.join(self.output_dir, "single_node_df.csv"), index=False)
        
    def plot_all_models(self, vllm_bench_csv=None, device=None, nodes_needed=1, target_qps_per_node=1.0):
        """Create plots for all models in the filtered data."""
        if self.filtered_df is None:
            print("No data loaded. Call load_and_filter_data() first.")
            return
            
        unique_models = self.filtered_df['model_name'].unique()
        
        for model_name in unique_models:
            print(f"\nProcessing model: {model_name}")
            model_data = self.filtered_df[self.filtered_df['model_name'] == model_name]
            
            # Create both QPS and QPS per dollar plots
            self.create_subplots_for_model(model_name, model_data, plot_type='qps')
            self.create_subplots_for_model(model_name, model_data, plot_type='qps_per_dollar')
        
        # Create the bar chart comparison
        self.plot_max_qps_per_dollar_barchart()
        
        # Create parallelism strategies plot
        network_devices = ["a10g_g5", "h100_p5", "l40s_g6e48", "p4d_a100_40g_nvlink", "l4_g6"]
        for device in network_devices:
            self.plot_parallelism_strategies(instance_type=device)
        
        # Create vLLM comparison if parameters provided
        if vllm_bench_csv and device and os.path.exists(vllm_bench_csv):
            self.plot_vidur_vs_vllm_comparison(vllm_bench_csv, device, nodes_needed)
        
        # Create min total cost plot with specified qps_per_node
        self.plot_min_total_cost_by_qps_per_node(target_qps_per_node=target_qps_per_node)
    
    def plot_vidur_vs_vllm_comparison(self, vllm_bench_csv, device, nodes_needed=1):
        """Plot comparison between Vidur predictions and actual vLLM benchmark results."""
        if self.filtered_df is None:
            print("No Vidur data loaded. Call load_and_filter_data() first.")
            return
            
        # Load vLLM benchmark data
        vllm_df = pd.read_csv(vllm_bench_csv)
        
        # Filter vLLM data for the specified device
        vllm_filtered = vllm_df[vllm_df['Device'] == device].copy()
        
        if len(vllm_filtered) == 0:
            print(f"No vLLM benchmark data found for device: {device}")
            return
            
        # Filter Vidur data for the specified device and nodes_needed
        vidur_filtered = self.filtered_df[
            (self.filtered_df['network_device'] == device) & 
            (self.filtered_df['nodes_needed'] == nodes_needed)
        ].copy()
        
        if len(vidur_filtered) == 0:
            print(f"No Vidur data found for device: {device} with nodes_needed: {nodes_needed}")
            return
            
        # Process each model separately
        unique_models = vidur_filtered['model_name'].unique()
        
        for model_name in unique_models:
            # Filter data for this model
            model_vidur = vidur_filtered[vidur_filtered['model_name'] == model_name]
            model_vllm = vllm_filtered[vllm_filtered['Model'] == model_name]
            
            if len(model_vllm) == 0:
                print(f"No vLLM data for model {model_name}")
                continue
                
            # Get unique QPS values from vLLM data for this model
            unique_qps = sorted(model_vllm['QPS'].unique())
            
            # Prepare data for plotting
            qps_values = []
            vllm_valid_configs = []
            pred_vs_worst_ratios = []
            best_vs_worst_ratios = []
            
            for qps in unique_qps:
                # Get vLLM results for this QPS
                vllm_qps_data = model_vllm[model_vllm['QPS'] == qps]
                
                if len(vllm_qps_data) == 0:
                    continue
                
                # Find best and worst P99 exec latency in vLLM data
                actual_best_latency = vllm_qps_data['Total_Gen_P99'].min()
                actual_worst_latency = vllm_qps_data['Total_Gen_P99'].max()
                
                # Find Vidur's predicted best config for this exact QPS
                viable_configs = model_vidur[model_vidur['qps'] == qps]
                
                if len(viable_configs) == 0:
                    continue
                    
                # Find config with minimum P99 exec latency among viable configs
                pred_best_config = viable_configs.loc[viable_configs['p99_exec_time'].idxmin()]
                
                # Get the predicted config parameters
                num_replicas = pred_best_config['num_replicas']
                tensor_parallel = pred_best_config['tensor_parallel_size']
                pipeline_parallel = pred_best_config['num_pipeline_stages']
                
                # Find matching vLLM result with same config
                vllm_match = vllm_qps_data[
                    (vllm_qps_data['Num_Replicas'] == num_replicas) &
                    (vllm_qps_data['Tensor_Parallel'] == tensor_parallel) &
                    (vllm_qps_data['Pipeline_Parallel'] == pipeline_parallel)
                ]
                
                if len(vllm_match) == 0:
                    print(f"no vllm match for qps={qps}, dp,tp,pp={num_replicas}, {tensor_parallel}, {pipeline_parallel}")
                    continue
                    
                pred_latency = vllm_match['Total_Gen_P99'].iloc[0]
                
                # Calculate ratios (lower is better for latency)
                if actual_worst_latency > 0:
                    pred_vs_worst_ratio = pred_latency / actual_worst_latency
                    best_vs_worst_ratio = actual_best_latency / actual_worst_latency
                    
                    qps_values.append(qps)
                    vllm_valid_configs.append(len(vllm_qps_data))
                    pred_vs_worst_ratios.append(pred_vs_worst_ratio)
                    best_vs_worst_ratios.append(best_vs_worst_ratio)
                    
            if not qps_values:
                print(f"No matching QPS values found for model {model_name}")
                continue
                
            # Create bar plot for this model
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x = np.arange(len(qps_values))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, pred_vs_worst_ratios, width, 
                          label='Vidur Predicted vs Actual Worst', alpha=0.7, color='skyblue')
            bars2 = ax.bar(x + width/2, best_vs_worst_ratios, width,
                          label='Actual Best vs Actual Worst', alpha=0.7, color='orange')
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=10)
                       
            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=10)
            
            # Formatting
            ax.set_xlabel('QPS', fontsize=14)
            ax.set_ylabel('Latency Ratio (P99 Exec Time)', fontsize=14)
            ax.set_title(f'Vidur vs vLLM Comparison - {model_name}\nDevice: {device}, Nodes: {nodes_needed}', 
                        fontsize=16, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f'{qps:.1f}' for qps in qps_values], rotation=45)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add horizontal line at y=1 for reference
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect Ratio')
            
            plt.tight_layout()
            
            # Save plot
            clean_model_name = model_name.replace('/', '_').replace(' ', '_')
            filename = f"vidur_vs_vllm_{clean_model_name}_{device}_nodes{nodes_needed}.png"
            filepath = os.path.join(self.output_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved comparison plot for {model_name}: {filepath}")
            
            # Save detailed data
            comparison_data = pd.DataFrame({
                'QPS': qps_values,
                "vllm_available_configs": vllm_valid_configs,
                'Vidur_vs_Worst_Ratio': pred_vs_worst_ratios,
                'Best_vs_Worst_Ratio': best_vs_worst_ratios
            })
            
            csv_filename = f"vidur_vs_vllm_{clean_model_name}_{device}_nodes{nodes_needed}_data.csv"
            csv_filepath = os.path.join(self.output_dir, csv_filename)
            comparison_data.to_csv(csv_filepath, index=False)
            print(f"Saved comparison data for {model_name}: {csv_filepath}")

    def plot_min_total_cost_by_qps_per_node(self, target_qps_per_node=1.0):
        """Create bar chart showing minimum total cost for each instance type at target QPS per node."""
        if self.filtered_df is None:
            print("No data loaded. Call load_and_filter_data() first.")
            return
            
        unique_models = self.filtered_df['model_name'].unique()
        n_models = len(unique_models)
        
        if n_models == 0:
            print("No models found in data.")
            return
        
        # Create subplots for each model
        n_cols = int(np.ceil(np.sqrt(n_models)))
        n_rows = int(np.ceil(n_models / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        # Process each model
        for i, model_name in enumerate(unique_models):
            ax = axes[i]
            model_data = self.filtered_df[self.filtered_df['model_name'] == model_name]
            
            # Apply SLO constraints
            slo_compliant = model_data[
                (model_data['p99_ttft'] <= self.slo_limit) & 
                (model_data['p99_exec_time'] <= self.exec_slo)
            ]
            
            if len(slo_compliant) == 0:
                ax.text(0.5, 0.5, f"No SLO compliant configs for {model_name}", ha='center', va='center')
                ax.set_title(f'{model_name} - Min Total Cost')
                continue
            
            # Calculate qps_per_node for each config
            slo_compliant = slo_compliant.copy()
            slo_compliant['qps_per_node'] = slo_compliant['qps'] / slo_compliant['nodes_needed']
            
            # Group by network_device and find config closest to target_qps_per_node with min cost
            device_data_dict = {}
            
            for device in slo_compliant['network_device'].unique():
                device_data = slo_compliant[slo_compliant['network_device'] == device]
                
                # Find configs with qps_per_node >= target_qps_per_node
                viable_configs = device_data[device_data['qps_per_node'] >= target_qps_per_node]
                
                if len(viable_configs) > 0:
                    # Among viable configs, find the one with minimum total cost
                    min_cost_config = viable_configs.loc[viable_configs['total_cost'].idxmin()]
                    device_data_dict[device] = {
                        'cost': min_cost_config['total_cost'],
                        'qps': min_cost_config['qps'],
                        'nodes': min_cost_config['nodes_needed']
                    }
            
            # Order instances and extract data
            instance_names = self._get_instance_order(list(device_data_dict.keys()))
            instance_min_costs = [device_data_dict[inst]['cost'] for inst in instance_names]
            selected_qps = [device_data_dict[inst]['qps'] for inst in instance_names]
            selected_nodes = [device_data_dict[inst]['nodes'] for inst in instance_names]
            
            if not instance_min_costs:
                ax.text(0.5, 0.5, f"No configs meet QPS/node >= {target_qps_per_node} for {model_name}", 
                       ha='center', va='center')
                ax.set_title(f'{model_name} - Min Total Cost')
                continue
            
            # Create bar plot
            bars = ax.bar(instance_names, instance_min_costs, alpha=0.7, color='gold', edgecolor='black')
            
            # Find minimum cost for percentage calculation
            min_cost = min(instance_min_costs)
            
            # Add percentage text boxes with QPS info at top
            for j, (instance, cost, qps) in enumerate(zip(instance_names, instance_min_costs, selected_qps)):
                if cost == min_cost:
                    percentage_text = '0%'
                else:
                    percentage_diff = ((cost - min_cost) / min_cost) * 100
                    percentage_text = f'+{percentage_diff:.1f}%'
                
                text_content = f'{percentage_text}\nQPS: {qps:.1f}'
                ax.text(j, cost + max(instance_min_costs) * 0.02, text_content, 
                       ha='center', va='bottom', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            # Add nodes info at bottom of each bar
            for j, nodes in enumerate(selected_nodes):
                ax.text(j, max(instance_min_costs) * 0.05, f'Node: {int(nodes)}', 
                       ha='center', va='bottom', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.7))
            
            # Formatting
            ax.set_title(f'{model_name} - Min Total Cost\n(QPS/Node >= {target_qps_per_node})', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Instance Type', fontsize=12)
            ax.set_ylabel('Min Total Cost ($)', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(instance_min_costs) * 1.2)
        
        # Hide unused subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        # Add overall title
        fig.suptitle(f'Minimum Total Cost by Instance Type\n(QPS per Node >= {target_qps_per_node}, SLO Compliant)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the plot
        filename = f"min_total_cost_qps_per_node_{target_qps_per_node}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved min total cost plot: {filepath}")

    def print_summary(self):
        """Print summary statistics of the filtered data."""
        if self.filtered_df is None:
            print("No data loaded.")
            return
            
        print("\n" + "="*50)
        print("DATA SUMMARY")
        print("="*50)
        print(f"Total configurations: {len(self.filtered_df)}")
        print(f"Input parameters: {self.input_param}")
        
        print(f"\nUnique models: {list(self.filtered_df['model_name'].unique())}")
        print(f"Unique network devices: {list(self.filtered_df['network_device'].unique())}")
        
        # Summary by model
        for model_name in self.filtered_df['model_name'].unique():
            model_data = self.filtered_df[self.filtered_df['model_name'] == model_name]
            max_data = self.get_max_qps_per_dollar_data(model_data)
            
            print(f"\n{model_name}:")
            print(f"  Total configs: {len(model_data)}")
            print(f"  Max QPS per dollar points: {len(max_data)}")
            if len(max_data) > 0:
                print(f"  Best QPS per dollar: {max_data['qps_per_dollar'].max():.4f}")
                print(f"  Best QPS: {max_data['qps'].max():.2f}")


def main():
    """Main function to run the plotter."""
    # Configuration
    csv_file = "config_optimizer_results.csv"
    input_param = {'prefill_tokens': 300, 'decode_tokens': 3}
    
    # Create plotter instance
    plotter = ConfigOptimizerPlotter(csv_file, input_param)
    
    # Load and filter data
    plotter.load_and_filter_data()
    
    # Print summary
    plotter.print_summary()
    
    # Create plots for all models with vLLM comparison
    vllm_bench_csv = "/home/ec2-user/vidur-simulator/analysis/vllm_bench_results/qu_brands/latency_metrics_20250819_191444.csv"
    device = "a10g_g5"  # or "p4d_a100_40g_nvlink"
    nodes_needed = 1
    
    plotter.plot_all_models(vllm_bench_csv, device, nodes_needed)
    
    print("\nPlotting complete!")


if __name__ == "__main__":
    main()
