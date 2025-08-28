# Model Benchmark Comparison Tool

This directory contains tools for benchmarking and comparing the performance of different LLM models using vLLM.

## Plot Model Comparison

The `plot_model_comparison.py` script allows you to compare benchmark results between two models, generating bar plots for various performance metrics with percentage differences.

### Features

- Compare two models across multiple performance metrics
- Show percentage differences between models in each plot
- Filter by specific tensor parallelism (TP), data parallelism (DP), and pipeline parallelism (PP) configurations
- Generate plots for all available configurations
- Create combined plots with all metrics in a single figure

### Metrics Compared

- `overall_throughput_tokens_per_sec`: Throughput in tokens per second (higher is better)
- `total_latency_p50_ms`: 50th percentile of total latency in milliseconds (lower is better)
- `total_latency_p99_ms`: 99th percentile of total latency in milliseconds (lower is better)
- `ttft_p50_ms`: 50th percentile of time to first token in milliseconds (lower is better)
- `ttft_p99_ms`: 99th percentile of time to first token in milliseconds (lower is better)
- `inter_token_p50_ms`: 50th percentile of inter-token latency in milliseconds (lower is better)
- `inter_token_p99_ms`: 99th percentile of inter-token latency in milliseconds (lower is better)

### Usage

```bash
python plot_model_comparison.py --model1 <model1_name> --model2 <model2_name> [options]
```

#### Required Arguments

- `--model1`: First model name (e.g., Qwen3-4B)
- `--model2`: Second model name (e.g., Qwen3-4B-AWQ)

#### Optional Arguments

- `--base-dir`: Base directory containing model benchmark results (default: benchmarks/llm/vllm/latency/sweep_configs)
- `--output-dir`: Directory to save the plots (default: benchmark_comparison_plots)
- `--tp`: Tensor Parallelism value to filter by (default: 1)
- `--dp`: Data Parallelism value to filter by (default: 1)
- `--pp`: Pipeline Parallelism value to filter by (default: 1)
- `--all-configs`: Plot all available configurations
- `--combined`: Create a combined plot with all metrics

### Examples

1. Compare two models with default configuration (TP=1, DP=1, PP=1):
```bash
python plot_model_comparison.py --model1 Qwen3-4B --model2 Qwen3-4B-AWQ
```

2. Compare two models with a specific configuration:
```bash
python plot_model_comparison.py --model1 Qwen3-4B --model2 Qwen3-4B-AWQ --tp 2 --dp 4 --pp 1
```

3. Compare two models across all available configurations:
```bash
python plot_model_comparison.py --model1 Qwen3-4B --model2 Qwen3-4B-AWQ --all-configs
```

4. Generate combined plots with all metrics:
```bash
python plot_model_comparison.py --model1 Qwen3-4B --model2 Qwen3-4B-AWQ --combined
```

5. Generate combined plots for all configurations:
```bash
python plot_model_comparison.py --model1 Qwen3-4B --model2 Qwen3-4B-AWQ --all-configs --combined
```

### Output

The script generates PNG files in the specified output directory. For each configuration, it creates:
- Individual plots for each metric
- A combined plot with all metrics (if `--combined` is specified)

The plots show the values for each model and the percentage difference between them, with color coding to indicate whether the difference is an improvement (green) or regression (red).
