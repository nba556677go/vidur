#!/usr/bin/env python3
"""
Custom version of the max QPS per dollar plotter with configurable parameters.
Usage:
    python plot_max_qps_per_dollar_custom.py --prefill_tokens 300 --decode_tokens 3
    python plot_max_qps_per_dollar_custom.py --prefill_tokens 2048 --decode_tokens 512
"""

import argparse
import os
from plot_max_qps_per_dollar import ConfigOptimizerPlotter


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Plot max QPS per dollar configurations')
    
    parser.add_argument('--csv_file', type=str, 
                       default='config_optimizer_results.csv',
                       help='Path to CSV file with config optimizer results')
    
    parser.add_argument('--prefill_tokens', type=int, default=300,
                       help='Filter by prefill tokens (default: 300)')
    
    parser.add_argument('--decode_tokens', type=int, default=3,
                       help='Filter by decode tokens (default: 3)')
    
    parser.add_argument('--model_name', type=str, default=None,
                       help='Filter by specific model name (optional)')
    
    parser.add_argument('--network_device', type=str, default=None,
                       help='Filter by specific network device (optional)')
    
    parser.add_argument('--output_dir', type=str, default='max_qps',
                       help='Output directory to save plots (default: max_qps)')
    
    parser.add_argument('--slo_limit', type=float, default=0.25,
                       help='SLO limit for TTFT in seconds (default: 0.25)')
    
    parser.add_argument('--exec_slo', type=float, default=7.8,
                       help='SLO limit for total execution time in seconds (default: 7.8)')
    
    parser.add_argument('--inter_token_slo', type=float, default=0.030,
                       help='SLO limit for inter-token latency in seconds (default: 0.015)')
    
    parser.add_argument('--vllm_bench_csv', type=str, 
                       default='/home/ec2-user/vidur-simulator/analysis/vllm_bench_results/qu_brands/latency_metrics_20250819_234737.csv',
                       help='Path to vLLM benchmark CSV file')
    
    parser.add_argument('--device', type=str, default='a10g_g5',
                       help='Device for vLLM comparison (default: a10g_g5)')
    
    parser.add_argument('--nodes_needed', type=int, default=1,
                       help='Number of nodes needed (default: 1)')
    
    parser.add_argument('--target_qps_per_node', type=float, default=40,
                       help='Target QPS per node for min cost analysis (default: 20)')
    
    return parser.parse_args()


def main():
    """Main function with customizable parameters."""
    args = parse_arguments()
    
    # Build input parameters dictionary
    input_param = {
        'prefill_tokens': args.prefill_tokens,
        'decode_tokens': args.decode_tokens
    }
    
    # Add optional filters
    if args.model_name:
        input_param['model_name'] = args.model_name
    if args.network_device:
        input_param['network_device'] = args.network_device
    
    print(f"Using CSV file: {args.csv_file}")
    print(f"Input parameters: {input_param}")
    print(f"Output directory: {args.output_dir}")
    
    # Create plotter instance
    plotter = ConfigOptimizerPlotter(args.csv_file, input_param, args.output_dir, 
                                   args.slo_limit, args.exec_slo, args.inter_token_slo)
    
    # Load and filter data
    plotter.load_and_filter_data()
    
    # Print summary
    plotter.print_summary()
    
    # Create plots for all models with vLLM comparison
    plotter.plot_all_models(args.vllm_bench_csv, args.device, args.nodes_needed, args.target_qps_per_node)
    
    print("\nPlotting complete!")


if __name__ == "__main__":
    main()
