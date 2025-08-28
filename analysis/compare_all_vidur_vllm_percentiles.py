#!/usr/bin/env python3
"""
Script to compare VLLM benchmark and Vidur simulation results for P50, P90, and P99 latencies.
It generates a CSV with the results and then creates plots.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

def plot_aggregated_p99_latency(aggregate_df, output_dir):
    """
    Creates an aggregated plot showing P99 latency comparison across all model-device combinations.
    """
    devices = aggregate_df['VLLM_Device'].unique()
    models = aggregate_df['Model'].unique()
    
    total_subplots = len(devices) * len(models)
    n_cols = len(devices)
    n_rows = len(models)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
    if total_subplots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle('P99 Latency Comparison Across All Model-Device Combinations', fontsize=16, fontweight='bold', y=0.995)
    
    subplot_idx = 0
    for model in models:
        for device in devices:
            ax = axes[subplot_idx]
            subset_df = aggregate_df[(aggregate_df['Model'] == model) & (aggregate_df['VLLM_Device'] == device)]
            
            if len(subset_df) == 0:
                ax.text(0.5, 0.5, f'No data for {model}\n{device}', ha='center', va='center')
                ax.set_title(f'{model} - {device}', fontsize=14, fontweight='bold')
                subplot_idx += 1
                continue
            
            x_labels = subset_df['QPS'].astype(str)
            x = np.arange(len(x_labels))
            bar_width = 0.25
            
            vllm_ratio = subset_df.get('VLLM_Latency_Ratio_P99', pd.Series(np.nan, index=subset_df.index))
            r1tp1pp1_ratio = subset_df.get('R1TP1PP1_vs_VLLM_Min_P99', pd.Series(np.nan, index=subset_df.index))
            random_ratio = subset_df.get('Random_Config_P99', pd.Series(np.nan, index=subset_df.index))
            
            bar1 = ax.bar(x - bar_width, vllm_ratio, bar_width, label='Vidur', alpha=0.7, color='skyblue', edgecolor='black')
            bar2 = ax.bar(x, r1tp1pp1_ratio, bar_width, label='R1TP1PP1', alpha=0.7, color='orange', edgecolor='black')
            bar3 = ax.bar(x + bar_width, random_ratio, bar_width, label='Random', alpha=0.7, color='lightcoral', edgecolor='black')
            
            for j in range(len(x)):
                if j < len(vllm_ratio) and pd.notna(vllm_ratio.iloc[j]):
                    ax.text(x[j] - bar_width, vllm_ratio.iloc[j] + (max(vllm_ratio.fillna(0)) * 0.02), 
                           f'{vllm_ratio.iloc[j]:.2f}', ha='center', va='bottom', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                if j < len(r1tp1pp1_ratio) and pd.notna(r1tp1pp1_ratio.iloc[j]):
                    ax.text(x[j], r1tp1pp1_ratio.iloc[j] + (max(r1tp1pp1_ratio.fillna(0)) * 0.02), 
                           f'{r1tp1pp1_ratio.iloc[j]:.2f}', ha='center', va='bottom', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                if j < len(random_ratio) and pd.notna(random_ratio.iloc[j]):
                    ax.text(x[j] + bar_width, random_ratio.iloc[j] + (max(random_ratio.fillna(0)) * 0.02), 
                           f'{random_ratio.iloc[j]:.2f}', ha='center', va='bottom', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            ax.set_title(f'{model} - {device}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Latency Ratio', fontsize=12)
            ax.set_xlabel('QPS', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels)
            ax.tick_params(axis='both', which='major', labelsize=12)
            if ax.get_ylim()[1] > 0:
                ax.set_ylim(top=ax.get_ylim()[1] * 1.25)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            subplot_idx += 1
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plot_path = os.path.join(output_dir, 'aggregated_p99_latency_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Aggregated P99 latency plot saved to: {plot_path}")

def plot_aggregated_prediction_error(aggregate_df, output_dir):
    """
    Creates an aggregated plot showing prediction error comparison across all model-device combinations.
    """
    devices = aggregate_df['VLLM_Device'].unique()
    models = aggregate_df['Model'].unique()
    
    total_subplots = len(devices) * len(models)
    n_cols = len(devices)
    n_rows = len(models)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
    if total_subplots == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle('Prediction Error Comparison Across All Model-Device Combinations', fontsize=16, fontweight='bold', y=0.995)
    
    subplot_idx = 0
    for model in models:
        for device in devices:
            ax = axes[subplot_idx]
            subset_df = aggregate_df[(aggregate_df['Model'] == model) & (aggregate_df['VLLM_Device'] == device)]
            
            if len(subset_df) == 0:
                ax.text(0.5, 0.5, f'No data for {model}\n{device}', ha='center', va='center')
                ax.set_title(f'{model} - {device}', fontsize=14, fontweight='bold')
                subplot_idx += 1
                continue
            
            x_labels = subset_df['QPS'].astype(str)
            x = np.arange(len(x_labels))
            bar_width = 0.25
            
            error_p50 = subset_df.get('Avg_Pred_Error_P50', pd.Series(np.nan, index=subset_df.index))
            error_p90 = subset_df.get('Avg_Pred_Error_P90', pd.Series(np.nan, index=subset_df.index))
            error_p99 = subset_df.get('Avg_Pred_Error_P99', pd.Series(np.nan, index=subset_df.index))
            
            bar1 = ax.bar(x - bar_width, error_p50, bar_width, label='P50 Error', alpha=0.7, color='skyblue', edgecolor='black')
            bar2 = ax.bar(x, error_p90, bar_width, label='P90 Error', alpha=0.7, color='orange', edgecolor='black')
            bar3 = ax.bar(x + bar_width, error_p99, bar_width, label='P99 Error', alpha=0.7, color='lightcoral', edgecolor='black')
            
            for j in range(len(x)):
                if j < len(error_p50) and pd.notna(error_p50.iloc[j]):
                    ax.text(x[j] - bar_width, error_p50.iloc[j] + (max(error_p50.fillna(0)) * 0.02), 
                           f'{error_p50.iloc[j]:.1f}%', ha='center', va='bottom', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                if j < len(error_p90) and pd.notna(error_p90.iloc[j]):
                    ax.text(x[j], error_p90.iloc[j] + (max(error_p90.fillna(0)) * 0.02), 
                           f'{error_p90.iloc[j]:.1f}%', ha='center', va='bottom', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                if j < len(error_p99) and pd.notna(error_p99.iloc[j]):
                    ax.text(x[j] + bar_width, error_p99.iloc[j] + (max(error_p99.fillna(0)) * 0.02), 
                           f'{error_p99.iloc[j]:.1f}%', ha='center', va='bottom', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            ax.set_title(f'{model} - {device}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Prediction Error (%)', fontsize=12)
            ax.set_xlabel('QPS', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.axhline(0, color='grey', linewidth=0.8)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')
            
            subplot_idx += 1
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plot_path = os.path.join(output_dir, 'aggregated_prediction_error_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Aggregated prediction error plot saved to: {plot_path}")

def plot_results(df, output_dir):
    """
    Generates and saves plots based on the comparison results.
    This function is robust to single QPS entries and logs warnings for non-finite values.

    Args:
        df (pd.DataFrame): The DataFrame containing the comparison results.
        output_dir (str): The directory where the plots will be saved.
    """
    print("\nGenerating plots...")
    
    plot_df = df.copy()
    x_labels = plot_df['QPS'].astype(str)
    x = np.arange(len(x_labels))
    num_qps = len(x_labels)

    # --- Plot 1: Latency Ratio Comparison ---
    fig, axes = plt.subplots(3, 1, figsize=(15, 22))
    fig.suptitle('Comparison of Latency Ratios for different QPS', fontsize=16, fontweight='bold', y=0.995)
    percentiles = ['P50', 'P90', 'P99']
    bar_width_latency = 0.8 / 3 if num_qps > 1 else 0.25
    bar_label_fontsize = 10 # Fontsize for the new labels

    for i, p in enumerate(percentiles):
        ax = axes[i]
        
        for col_name_template in ['VLLM_Latency_Ratio_{}', 'R1TP1PP1_vs_VLLM_Min_{}']:
            col_name = col_name_template.format(p)
            if col_name in plot_df.columns:
                plot_df[col_name] = pd.to_numeric(plot_df[col_name], errors='coerce')
                if not np.isfinite(plot_df[col_name]).all():
                    invalid_qps = plot_df['QPS'][~np.isfinite(plot_df[col_name])]
                    warnings.warn(f"\n[!] Warning: Non-finite values in '{col_name}' for QPS: {list(invalid_qps)}. Skipped in plot.")
            else:
                warnings.warn(f"\n[!] Warning: Column '{col_name}' not found. Skipping.")
                continue

        vllm_latency_ratio = plot_df.get(f'VLLM_Latency_Ratio_{p}', pd.Series(np.nan, index=plot_df.index))
        r1tp1pp1_vs_vllm_min = plot_df.get(f'R1TP1PP1_vs_VLLM_Min_{p}', pd.Series(np.nan, index=plot_df.index))
        random_config = plot_df.get(f'Random_Config_{p}', pd.Series(np.nan, index=plot_df.index))

        bar1 = ax.bar(x - bar_width_latency, vllm_latency_ratio, bar_width_latency, label='Vidur', alpha=0.7, color='skyblue', edgecolor='black')
        bar2 = ax.bar(x, r1tp1pp1_vs_vllm_min, bar_width_latency, label='R1TP1PP1', alpha=0.7, color='orange', edgecolor='black')
        bar3 = ax.bar(x + bar_width_latency, random_config, bar_width_latency, label='Random', alpha=0.7, color='lightcoral', edgecolor='black')

        # --- ADDED THIS SECTION FOR LABELS ---
        # Add custom text labels with styled boxes for latency ratios
        for j in range(len(x)):
            if j < len(vllm_latency_ratio) and pd.notna(vllm_latency_ratio.iloc[j]):
                ax.text(x[j] - bar_width_latency, vllm_latency_ratio.iloc[j] + (max(vllm_latency_ratio.fillna(0)) * 0.02 if not vllm_latency_ratio.empty else 0), 
                       f'{vllm_latency_ratio.iloc[j]:.2f}', ha='center', va='bottom', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            if j < len(r1tp1pp1_vs_vllm_min) and pd.notna(r1tp1pp1_vs_vllm_min.iloc[j]):
                ax.text(x[j], r1tp1pp1_vs_vllm_min.iloc[j] + (max(r1tp1pp1_vs_vllm_min.fillna(0)) * 0.02 if not r1tp1pp1_vs_vllm_min.empty else 0), 
                       f'{r1tp1pp1_vs_vllm_min.iloc[j]:.2f}', ha='center', va='bottom', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            if j < len(random_config) and pd.notna(random_config.iloc[j]):
                ax.text(x[j] + bar_width_latency, random_config.iloc[j] + (max(random_config.fillna(0)) * 0.02 if not random_config.empty else 0), 
                       f'{random_config.iloc[j]:.2f}', ha='center', va='bottom', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        # --- END OF ADDITION ---

        ax.set_title(f'{p} Latency Comparison - {plot_df["Model"].iloc[0] if not plot_df.empty else "Unknown Model"} - {plot_df["VLLM_Device"].iloc[0] if not plot_df.empty else "Unknown Device"}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Latency Ratio', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.tick_params(axis='both', which='major', labelsize=12)
        if num_qps == 1:
            ax.set_xlim(-1, 1)
        if ax.get_ylim()[1] > 0:
            ax.set_ylim(top=ax.get_ylim()[1] * 1.25) # Increased padding for labels
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

    plt.xlabel('QPS (Queries Per Second)', fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    latency_plot_path = os.path.join(output_dir, 'latency_comparison_by_qps_detailed.png')
    plt.savefig(latency_plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Latency comparison plot saved to: {latency_plot_path}")

    # --- Plot 2: Prediction Error Comparison ---
    fig, ax = plt.subplots(figsize=(14, 9))
    bar_width_error = 0.25

    error_cols = ['Avg_Pred_Error_P50', 'Avg_Pred_Error_P90', 'Avg_Pred_Error_P99']
    for col_name in error_cols:
        if col_name in plot_df.columns:
            plot_df[col_name] = pd.to_numeric(plot_df[col_name], errors='coerce')
            if not np.isfinite(plot_df[col_name]).all():
                invalid_qps = plot_df['QPS'][~np.isfinite(plot_df[col_name])]
                warnings.warn(f"\n[!] Warning: Non-finite values in '{col_name}' for QPS: {list(invalid_qps)}. Skipped in plot.")
        else:
             warnings.warn(f"\n[!] Warning: Column '{col_name}' not found. Skipping.")

    error_p50 = plot_df.get('Avg_Pred_Error_P50', pd.Series(np.nan, index=plot_df.index))
    error_p90 = plot_df.get('Avg_Pred_Error_P90', pd.Series(np.nan, index=plot_df.index))
    error_p99 = plot_df.get('Avg_Pred_Error_P99', pd.Series(np.nan, index=plot_df.index))
    
    bar1 = ax.bar(x - bar_width_error, error_p50, bar_width_error, label='P50 Error', alpha=0.7, color='skyblue', edgecolor='black')
    bar2 = ax.bar(x, error_p90, bar_width_error, label='P90 Error', alpha=0.7, color='orange', edgecolor='black')
    bar3 = ax.bar(x + bar_width_error, error_p99, bar_width_error, label='P99 Error', alpha=0.7, color='lightcoral', edgecolor='black')

    # Add custom text labels with styled boxes
    for i, (p50, p90, p99) in enumerate(zip(error_p50, error_p90, error_p99)):
        if pd.notna(p50):
            ax.text(i - bar_width_error, p50 + (max(error_p50.fillna(0)) * 0.02 if not error_p50.empty else 0), f'{p50:.1f}%', 
                   ha='center', va='bottom', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        if pd.notna(p90):
            ax.text(i, p90 + (max(error_p90.fillna(0)) * 0.02 if not error_p90.empty else 0), f'{p90:.1f}%', 
                   ha='center', va='bottom', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        if pd.notna(p99):
            ax.text(i + bar_width_error, p99 + (max(error_p99.fillna(0)) * 0.02 if not error_p99.empty else 0), f'{p99:.1f}%', 
                   ha='center', va='bottom', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    ax.set_title(f'Average Prediction Error by QPS - {plot_df["Model"].iloc[0] if not plot_df.empty else "Unknown Model"} - {plot_df["VLLM_Device"].iloc[0] if not plot_df.empty else "Unknown Device"}', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('QPS (Queries Per Second)', fontsize=12)
    ax.set_ylabel('Average Prediction Error (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.axhline(0, color='grey', linewidth=0.8)
    
    valid_errors = plot_df[error_cols].replace([np.inf, -np.inf], np.nan).dropna()
    if not valid_errors.empty:
        min_val = valid_errors.min().min()
        max_val = valid_errors.max().max()
        bottom_limit = min(0, min_val * 1.1)
        top_limit = max(0, max_val * 1.1)
        ax.set_ylim(bottom_limit, top_limit)
    
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    error_plot_path = os.path.join(output_dir, 'prediction_error_by_qps.png')
    plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Prediction error plot saved to: {error_plot_path}")


def main():
    # This main function is IDENTICAL to your original script
    vidur_profile = "qu_brand/fixed_qps"
    #profile list = (vidur_compute_profile, vidur_network_device, vllm_device)
    profile_list = [("h100_p5", "h100_p5", "h100_p5"), ("l40s_g6e48", "l40s_g6e48", "l40s_g6e48"), ("a100_p4d", "p4d_a100_40g_nvlink", "a100_p4d"),
                    ("a10g_g5", "a10g_g5", "a10g_g5") , ("l4_g6", "l4_g6", "l4_g6")] 
    #profile_list = [ ("a100_p4d", "p4d_a100_40g_nvlink", "a100_p4d") ]
    vidur_root_dir = f"vidur_results/{vidur_profile}"
    models = ["Qwen/Qwen2.5-1.5B", "meta-llama/Meta-Llama-3-8B"]
    #models = ["Qwen/Qwen2.5-1.5B"]
    aggregate_results = []
    for model in models:
        for profile  in profile_list:
            vidur_compute_profile, vidur_network_device, vllm_device = profile
            vidur_base_dir = f"{vidur_root_dir}/compute_{vidur_compute_profile}/network_{vidur_network_device}/{model}/chunk8192"
            vllm_bench_base_dir = f"vllm_bench_results/qu_brand/fixed_qps/{vllm_device}/nprompt300"
            
            qps_values = [4, 8, 10, 29,40 , 50, 60]
            #qps_values = [29,40 , 50, 60]
            #map qps with float
            qps_dir_map = {qps: str(qps) for qps in qps_values}
            #append qps_values with 2,5,8,..., 2.0,5.0,8.0... based on qps_values. d
            
            vllm_qps_dir_map = { 0.25: "0.25", 0.5: "0.5", 2: "2", 5:"5",  8: "8", 15: "15", 25: "25" }
            vidur_qps_dir_map = { 0.25: "0.25", 0.5: "0.5", 2: "2.0", 5:"5.0", 8: "8.0", 15: "15.0", 25: "25.0" }
            
            base_dir = os.path.dirname(os.path.abspath(__file__))
            results = []
            print(f"processing vidur path={vidur_base_dir}, vllm dir={vllm_bench_base_dir}")
            for qps in qps_values:
                vllm_qps_dir = qps_dir_map.get(qps, str(qps))
                vidur_qps_dir = qps_dir_map.get(qps, str(qps))
                #vllm_qps_dir = vllm_qps_dir_map.get(qps, str(qps))
                #vidur_qps_dir = vidur_qps_dir_map.get(qps, str(qps))
                
                vllm_dir = os.path.join(base_dir, f"{vllm_bench_base_dir}/qps{vllm_qps_dir}")
                if not os.path.exists(vllm_dir):
                    print(f"Directory not found: {vllm_dir}")
                    continue
                    
                vllm_files = [f for f in os.listdir(vllm_dir) if f.startswith("summary_")]
                if not vllm_files:
                    print(f"No VLLM summary files found in {vllm_dir}")
                    continue
                vllm_path = os.path.join(vllm_dir, vllm_files[0])
                vllm_df = pd.read_csv(vllm_path)
                vllm_df = vllm_df[vllm_df['Model'] == model]
                if len(vllm_df) == 0:
                    print(f"no vllm file for model = {model}")
                    continue
                vidur_dir = os.path.join(base_dir, f"{vidur_base_dir}/qps{vidur_qps_dir}")
                if not os.path.exists(vidur_dir):
                    print(f"Directory not found: {vidur_dir}")
                    continue
                    
                vidur_files = [f for f in os.listdir(vidur_dir) if f.startswith("summary_")]
                if not vidur_files:
                    print(f"No Vidur summary files found in {vidur_dir}")
                    continue
                
                vidur_path = os.path.join(vidur_dir, vidur_files[0])
                vidur_df = pd.read_csv(vidur_path)
                
                vllm_min_p50_config = vllm_df.loc[vllm_df['Total_Gen_P50'].idxmin()]
                vllm_min_p90_config = vllm_df.loc[vllm_df['Total_Gen_P90'].idxmin()]
                vllm_min_p99_config = vllm_df.loc[vllm_df['Total_Gen_P99'].idxmin()]
                
                vidur_min_p50_config = vidur_df.loc[vidur_df['Exec_P50'].idxmin()]
                vidur_min_p90_config = vidur_df.loc[vidur_df['Exec_P90'].idxmin()]
                vidur_min_p99_config = vidur_df.loc[vidur_df['Exec_P99'].idxmin()]
                
                vllm_with_vidur_p50_config = vllm_df[(vllm_df['Num_Replicas'] == vidur_min_p50_config['Num_Replicas']) & (vllm_df['Tensor_Parallel'] == vidur_min_p50_config['Tensor_Parallel']) & (vllm_df['Pipeline_Parallel'] == vidur_min_p50_config['Pipeline_Parallel'])]
                vllm_with_vidur_p90_config = vllm_df[(vllm_df['Num_Replicas'] == vidur_min_p90_config['Num_Replicas']) & (vllm_df['Tensor_Parallel'] == vidur_min_p90_config['Tensor_Parallel']) & (vllm_df['Pipeline_Parallel'] == vidur_min_p90_config['Pipeline_Parallel'])]
                vllm_with_vidur_p99_config = vllm_df[(vllm_df['Num_Replicas'] == vidur_min_p99_config['Num_Replicas']) & (vllm_df['Tensor_Parallel'] == vidur_min_p99_config['Tensor_Parallel']) & (vllm_df['Pipeline_Parallel'] == vidur_min_p99_config['Pipeline_Parallel'])]
                
                vllm_using_vidur_p50_config_p50 = np.inf
                vllm_using_vidur_p90_config_p90 = np.inf
                vllm_using_vidur_p99_config_p99 = np.inf
                
                if not vllm_with_vidur_p50_config.empty:
                    vllm_using_vidur_p50_config_p50 = vllm_with_vidur_p50_config['Total_Gen_P50'].values[0] / 1000.0
                if not vllm_with_vidur_p90_config.empty:
                    vllm_using_vidur_p90_config_p90 = vllm_with_vidur_p90_config['Total_Gen_P90'].values[0] / 1000.0
                if not vllm_with_vidur_p99_config.empty:
                    vllm_using_vidur_p99_config_p99 = vllm_with_vidur_p99_config['Total_Gen_P99'].values[0] / 1000.0
                    
                vllm_r1_tp1_pp1 = vllm_df[(vllm_df['Num_Replicas'] == 1) & (vllm_df['Tensor_Parallel'] == 1) & (vllm_df['Pipeline_Parallel'] == 1)]
                
                vllm_r1_tp1_pp1_p50, vllm_r1_tp1_pp1_p90, vllm_r1_tp1_pp1_p99 = np.inf, np.inf, np.inf
                if not vllm_r1_tp1_pp1.empty:
                    vllm_r1_tp1_pp1_p50 = vllm_r1_tp1_pp1['Total_Gen_P50'].values[0] / 1000.0
                    vllm_r1_tp1_pp1_p90 = vllm_r1_tp1_pp1['Total_Gen_P90'].values[0] / 1000.0
                    vllm_r1_tp1_pp1_p99 = vllm_r1_tp1_pp1['Total_Gen_P99'].values[0] / 1000.0
                
                vllm_p50 = vllm_min_p50_config['Total_Gen_P50'] / 1000.0
                vllm_p90 = vllm_min_p90_config['Total_Gen_P90'] / 1000.0
                vllm_p99 = vllm_min_p99_config['Total_Gen_P99'] / 1000.0
                
                vidur_p50 = vidur_min_p50_config['Exec_P50']
                vidur_p90 = vidur_min_p90_config['Exec_P90']
                vidur_p99 = vidur_min_p99_config['Exec_P99']
                
                vllm_latency_ratio_p50 = vllm_using_vidur_p50_config_p50 / vllm_p50 if vllm_p50 > 0 else np.inf
                vllm_latency_ratio_p90 = vllm_using_vidur_p90_config_p90 / vllm_p90 if vllm_p90 > 0 else np.inf
                vllm_latency_ratio_p99 = vllm_using_vidur_p99_config_p99 / vllm_p99 if vllm_p99 > 0 else np.inf
                
                r1_tp1_pp1_vs_vllm_min_p50 = vllm_r1_tp1_pp1_p50 / vllm_p50 if vllm_p50 > 0 else np.inf
                r1_tp1_pp1_vs_vllm_min_p90 = vllm_r1_tp1_pp1_p90 / vllm_p90 if vllm_p90 > 0 else np.inf
                r1_tp1_pp1_vs_vllm_min_p99 = vllm_r1_tp1_pp1_p99 / vllm_p99 if vllm_p99 > 0 else np.inf
                
                random_config_row = vllm_df.sample(n=1).iloc[0]
                random_config_p50 = (random_config_row['Total_Gen_P50'] / 1000.0) / vllm_p50 if vllm_p50 > 0 else np.inf
                random_config_p90 = (random_config_row['Total_Gen_P90'] / 1000.0) / vllm_p90 if vllm_p90 > 0 else np.inf
                random_config_p99 = (random_config_row['Total_Gen_P99'] / 1000.0) / vllm_p99 if vllm_p99 > 0 else np.inf
                
                merged_df = pd.merge(vllm_df, vidur_df, on=['Num_Replicas', 'Tensor_Parallel', 'Pipeline_Parallel'], suffixes=('_vllm', '_vidur'))
                avg_pred_error_p50, avg_pred_error_p90, avg_pred_error_p99 = np.inf, np.inf, np.inf
                if not merged_df.empty:
                    p50_error = ((merged_df['Exec_P50'] - (merged_df['Total_Gen_P50'] / 1000)) / (merged_df['Total_Gen_P50'] / 1000)) * 100
                    p90_error = ((merged_df['Exec_P90'] - (merged_df['Total_Gen_P90'] / 1000)) / (merged_df['Total_Gen_P90'] / 1000)) * 100
                    p99_error = ((merged_df['Exec_P99'] - (merged_df['Total_Gen_P99'] / 1000)) / (merged_df['Total_Gen_P99'] / 1000)) * 100
                    avg_pred_error_p50 = p50_error.mean()
                    avg_pred_error_p90 = p90_error.mean()
                    avg_pred_error_p99 = p99_error.mean()
                
                result_entry = {
                    'Model': model,
                    'Compute_Profile': vidur_compute_profile,
                    'Network_Device': vidur_network_device,
                    'VLLM_Device': vllm_device,
                    'QPS': qps,
                    'VLLM_P50_Config': f"R{int(vllm_min_p50_config['Num_Replicas'])}-TP{int(vllm_min_p50_config['Tensor_Parallel'])}-PP{int(vllm_min_p50_config['Pipeline_Parallel'])}",
                    'VLLM_P90_Config': f"R{int(vllm_min_p90_config['Num_Replicas'])}-TP{int(vllm_min_p90_config['Tensor_Parallel'])}-PP{int(vllm_min_p90_config['Pipeline_Parallel'])}",
                    'VLLM_P99_Config': f"R{int(vllm_min_p99_config['Num_Replicas'])}-TP{int(vllm_min_p99_config['Tensor_Parallel'])}-PP{int(vllm_min_p99_config['Pipeline_Parallel'])}",
                    'VLLM_P50': vllm_p50, 'VLLM_P90': vllm_p90, 'VLLM_P99': vllm_p99,
                    'Vidur_P50_Config': f"R{int(vidur_min_p50_config['Num_Replicas'])}-TP{int(vidur_min_p50_config['Tensor_Parallel'])}-PP{int(vidur_min_p50_config['Pipeline_Parallel'])}",
                    'Vidur_P90_Config': f"R{int(vidur_min_p90_config['Num_Replicas'])}-TP{int(vidur_min_p90_config['Tensor_Parallel'])}-PP{int(vidur_min_p90_config['Pipeline_Parallel'])}",
                    'Vidur_P99_Config': f"R{int(vidur_min_p99_config['Num_Replicas'])}-TP{int(vidur_min_p99_config['Tensor_Parallel'])}-PP{int(vidur_min_p99_config['Pipeline_Parallel'])}",
                    'Vidur_P50': vidur_p50, 'Vidur_P90': vidur_p90, 'Vidur_P99': vidur_p99,
                    'Ratio_P50': vllm_p50 / vidur_p50 if vidur_p50 > 0 else np.inf,
                    'Ratio_P90': vllm_p90 / vidur_p90 if vidur_p90 > 0 else np.inf,
                    'Ratio_P99': vllm_p99 / vidur_p99 if vidur_p99 > 0 else np.inf,
                    'VLLM_Latency_Ratio_P50': vllm_latency_ratio_p50,
                    'VLLM_Latency_Ratio_P90': vllm_latency_ratio_p90,
                    'VLLM_Latency_Ratio_P99': vllm_latency_ratio_p99,
                    'R1TP1PP1_vs_VLLM_Min_P50': r1_tp1_pp1_vs_vllm_min_p50,
                    'R1TP1PP1_vs_VLLM_Min_P90': r1_tp1_pp1_vs_vllm_min_p90,
                    'R1TP1PP1_vs_VLLM_Min_P99': r1_tp1_pp1_vs_vllm_min_p99,
                    'Random_Config_P50': random_config_p50,
                    'Random_Config_P90': random_config_p90,
                    'Random_Config_P99': random_config_p99,
                    'Avg_Pred_Error_P50': avg_pred_error_p50,
                    'Avg_Pred_Error_P90': avg_pred_error_p90,
                    'Avg_Pred_Error_P99': avg_pred_error_p99
                }
                results.append(result_entry)
                aggregate_results.append(result_entry)
            
            if not results:
                print("No results were generated. Exiting.")
                return

            results_df = pd.DataFrame(results)
            print("\nResults:")
            print(results_df.to_string(index=False, float_format='%.6f'))
            
            os.makedirs(vidur_base_dir, exist_ok=True)
            output_path = f"{vidur_base_dir}/all_percentiles_comparison.csv"
            results_df.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")

            if not results_df.empty:
                plot_results(results_df, vidur_base_dir)
    
    if aggregate_results:
        aggregate_df = pd.DataFrame(aggregate_results)
        os.makedirs(vidur_root_dir, exist_ok=True)
        aggregate_path = f"{vidur_root_dir}/aggregate_prediction_errors.csv"
        aggregate_df.to_csv(aggregate_path, index=False)
        print(f"\nAggregate prediction errors saved to: {aggregate_path}")
        
        # Create aggregated P99 latency plot
        plot_aggregated_p99_latency(aggregate_df, vidur_root_dir)
        
        # Create aggregated prediction error plot
        plot_aggregated_prediction_error(aggregate_df, vidur_root_dir)

if __name__ == "__main__":
    main()
