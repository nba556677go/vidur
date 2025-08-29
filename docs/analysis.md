
# Analyze Simulation and Benchmarked Data

## ⚠️ Warning
The commands below are examples based on previous benchmarked results stored in S3. Please reproduce first and verify input/output directory before executing. See [docs/reproduce.md](docs/reproduce.md)

## Move to Analysis Directory
```bash
cd analysis
```

## Update Unit Cost
Update `unitcost.csv` to match current instance.

## Vidur Parsing
```bash
python vidurparser.py
```

## VLLM Benchmark Parsing
Check input/output path in main before executing:

```bash
# Fixed QPS parsing
python vllmbenchparser_fixedqps.py

# QPS search mode parsing
python vllmbenchparser_qps_search.py
```
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

- [Cost analysis with parallelism strategies ]()


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
- [Cost analysis by parallelism]()
- Cost analysis by instance with benchmarks

#### Multinode Simulation

##### Figure 4: All-in-One Scatter Plot
- [Scatter plot]()

##### Figure 5: Cost Analysis
- [Multinode cost analysis]()

##### Figure 6: Cost by Instance with QPS Threshold
- [Cost analysis with QPS threshold]()

