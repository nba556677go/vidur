#!/bin/bash
#p4d
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device a100_p4d  --network_device p4d_a100_40g_nvlink --qps 29 --model Qwen/Qwen2.5-1.5B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device a100_p4d  --network_device p4d_a100_40g_nvlink --qps 40 --model Qwen/Qwen2.5-1.5B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device a100_p4d  --network_device p4d_a100_40g_nvlink --qps 50 --model Qwen/Qwen2.5-1.5B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device a100_p4d  --network_device p4d_a100_40g_nvlink --qps 60 --model Qwen/Qwen2.5-1.5B

python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device a100_p4d --network_device p4d_a100_40g_nvlink --qps 4 --model meta-llama/Meta-Llama-3-8B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device a100_p4d --network_device p4d_a100_40g_nvlink --qps 8 --model meta-llama/Meta-Llama-3-8B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device a100_p4d  --network_device p4d_a100_40g_nvlink --qps 10 --model meta-llama/Meta-Llama-3-8B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device a100_p4d  --network_device p4d_a100_40g_nvlink --qps 15 --model meta-llama/Meta-Llama-3-8B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device a100_p4d  --network_device p4d_a100_40g_nvlink --qps 29 --model meta-llama/Meta-Llama-3-8B

#p5
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device h100_p5  --network_device h100_p5 --qps 29 --model Qwen/Qwen2.5-1.5B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device h100_p5  --network_device h100_p5 --qps 40 --model Qwen/Qwen2.5-1.5B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device h100_p5  --network_device h100_p5 --qps 50 --model Qwen/Qwen2.5-1.5B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device h100_p5  --network_device h100_p5 --qps 60 --model Qwen/Qwen2.5-1.5B

python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device h100_p5 --network_device h100_p5 --qps 4 --model meta-llama/Meta-Llama-3-8B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device h100_p5  --network_device h100_p5 --qps 8 --model meta-llama/Meta-Llama-3-8B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device h100_p5  --network_device h100_p5 --qps 10 --model meta-llama/Meta-Llama-3-8B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device h100_p5  --network_device h100_p5 --qps 15 --model meta-llama/Meta-Llama-3-8B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device h100_p5  --network_device h100_p5 --qps 29 --model meta-llama/Meta-Llama-3-8B

#g6
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device l4_g6 --network_device l4_g6 --qps 29 --model Qwen/Qwen2.5-1.5B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device l4_g6  --network_device l4_g6 --qps 40 --model Qwen/Qwen2.5-1.5B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device l4_g6  --network_device l4_g6 --qps 50 --model Qwen/Qwen2.5-1.5B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device l4_g6  --network_device l4_g6 --qps 60 --model Qwen/Qwen2.5-1.5B

python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device l4_g6 --network_device l4_g6 --qps 4 --model meta-llama/Meta-Llama-3-8B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device l4_g6  --network_device l4_g6 --qps 8 --model meta-llama/Meta-Llama-3-8B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device l4_g6 --network_device l4_g6 --qps 10 --model meta-llama/Meta-Llama-3-8B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device l4_g6  --network_device l4_g6 --qps 15 --model meta-llama/Meta-Llama-3-8B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device l4_g6  --network_device l4_g6 --qps 29 --model meta-llama/Meta-Llama-3-8B

#g5
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device a10g_g5 --network_device a10g_g5 --qps 29 --model Qwen/Qwen2.5-1.5B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device a10g_g5  --network_device a10g_g5 --qps 40 --model Qwen/Qwen2.5-1.5B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device a10g_g5  --network_device a10g_g5 --qps 50 --model Qwen/Qwen2.5-1.5B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device a10g_g5  --network_device a10g_g5 --qps 60 --model Qwen/Qwen2.5-1.5B

python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device a10g_g5 --network_device a10g_g5 --qps 4 --model meta-llama/Meta-Llama-3-8B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device a10g_g5  --network_device a10g_g5 --qps 8 --model meta-llama/Meta-Llama-3-8B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device a10g_g5 --network_device a10g_g5 --qps 10 --model meta-llama/Meta-Llama-3-8B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device a10g_g5  --network_device a10g_g5 --qps 15 --model meta-llama/Meta-Llama-3-8B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device a10g_g5 --network_device a10g_g5 --qps 29 --model meta-llama/Meta-Llama-3-8B
#g6e
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device l40s_g6e48 --network_device l40s_g6e48 --qps 29 --model Qwen/Qwen2.5-1.5B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device l40s_g6e48  --network_device l40s_g6e48 --qps 40 --model Qwen/Qwen2.5-1.5B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device l40s_g6e48  --network_device l40s_g6e48 --qps 50 --model Qwen/Qwen2.5-1.5B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device l40s_g6e48  --network_device l40s_g6e48 --qps 60 --model Qwen/Qwen2.5-1.5B

python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device l40s_g6e48 --network_device l40s_g6e48 --qps 4 --model meta-llama/Meta-Llama-3-8B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device l40s_g6e48  --network_device l40s_g6e48 --qps 8 --model meta-llama/Meta-Llama-3-8B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device l40s_g6e48 --network_device l40s_g6e48 --qps 10 --model meta-llama/Meta-Llama-3-8B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device l40s_g6e48  --network_device l40s_g6e48 --qps 15 --model meta-llama/Meta-Llama-3-8B
python scripts/run_simulation.py  --log_dir simulator_output/qu_brand/   --replica_config_device l40s_g6e48 --network_device l40s_g6e48 --qps 29 --model meta-llama/Meta-Llama-3-8B


