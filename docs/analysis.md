
# Analyze Simulation and Benchmarked Data

## ⚠️ Warning
The commands below are examples based on previous runs stored in `analysis/vllm_benchresults` and `analysis/vidur_results`. Cost analysis plotted with synthetic cost in `analysis/unitcost.csv`. Please reproduce first and verify input/output directory before executing. See [reproduce.md](reproduce.md).

## Move to Analysis Directory
```bash
cd analysis
```

## Update Unit Cost
Update `unitcost.csv` to match actual cost.


## Single Node Fixed QPS Analysis

### Figures 1 & 2: Single Node Latency and Prediction Error Plots
```bash
python compare_all_vidur_vllm_percentiles.py
```

- [Latency comparison](../analysis/vidur_results/qu_brand/fixed_qps/aggregated_p99_latency_comparison.png)
- [Prediction error](../analysis/vidur_results/qu_brand/fixed_qps/aggregated_prediction_error_comparison.png)

### Figure 3: Parallelism Cost analysis
```bash
python get_config_optimizer_results_fixedqps.py
```

- [Cost analysis with parallelism strategies](../analysis/vidur_results/qu_brand/fixed_qps/parallel_figs/parallelism_strategies_a10g_g5_qps10.0.png)


## QPS Search Mode

### Parse All Vidur Search Data
Parse all vidur search data to `config_optimizer_results.csv`:

```bash
cd qps_search
python get_config_optimizer_results_qps_search.py
```

### Plotting
```bash
python plot_max_qps_per_dollar_main.py
```

#### Single Node Results with Benchmarks
- [Cost analysis by parallelism](../analysis/qps_search/max_qps/parallel_figs/parallelism_strategies_l40s_g6e48.png)
- [Cost analysis by instance with benchmarks](../analysis/qps_search/max_qps/max_qps_per_dollar_barchart_single_node.png)

#### Multinode Simulation

##### Figure 4: All-in-One Scatter Plot
- [Scatter plot](../analysis/qps_search/max_qps/max_qps_per_dollar_qps_Qwen_Qwen2.5-1.5B.png)

##### Figure 5: Cost Analysis
- [Multinode cost analysis](../analysis/qps_search/max_qps/max_qps_per_dollar_barchart_multinode.png)

##### Figure 6: Cost by Instance with QPS Threshold
- [Cost analysis with QPS threshold](../analysis/qps_search/max_qps/min_total_cost_qps_per_node_40.png)

