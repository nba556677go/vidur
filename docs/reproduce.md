# How to reproduce? 

## add model profiles stored in s3
```
aws s3 cp s3://magnus-test/binghann_vidur vidur_data --recursive
```

### unzip the profiles.zip into profile directory
```
unzip  vidur_data/profiling/profiles.zip -d data/profiling/
rm -f vidur_data/profiling/profiles.zip
```
## Simulation
### Fixed QPS simulation
```
bash scripts/run_simulation.sh
```
### QPS search mode with config-optimizer (explanation in [here](docs/config_explorer.md))
* comment each cluster info in config file if only desinated instance to be run
```
clusters:
  #- device: h100_p5
  #  network_device: h100_p5
  #  gpus_per_node: 8
  #- device: a100_p4d
  #  network_device: p4d_a100_40g_nvlink
  #  gpus_per_node: 8
  - device: l40s_g6e48
    network_device: l40s_g6e48
    gpus_per_node: 8
  # device: a10g_g5
  #  network_device: a10g_g5
  #  gpus_per_node: 8
  #- device: l4_g6
  #  network_device: l4_g6
  #  gpus_per_node: 8
```
#### Run command
```
python -u -m vidur.config_optimizer.config_explorer.main \
--config-path vidur/config_optimizer/config_explorer/config/config_aws.yml \
--cache-dir cache \
--output-dir config_optimizer_output \
--time-limit 180
```

## benchmarking
create vllm benchmark environment for ground truth comparison. 
* export huggingface token using export
```
export HF_TOKEN=<your_huggingface_token>
```
* build image and run
```
cd benchmark/llm/vllm
#build vllm image
docker build -t vllm:dev . 
#assume the repo is installed under $HOME
docker run -it --gpus all --ipc=host -v  $HOME/vidur-simulator:$HOME/vidur-simulator/ -v ~/.cache/huggingface:/root/.cache/huggingface -e RAY_CGRAPH_get_timeout=300 -e RAY_CGRAPH_submit_timeout=300 -e HF_TOKEN vllm:dev bash
```

* run benchmark script. change parameters within the script
```
cd  $HOME/vidur-simulator/benchmarks/llm/vllm/latency/
bash run_bench_latency.sh 
```

