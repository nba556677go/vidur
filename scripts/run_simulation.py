import subprocess
import os
import argparse
import logging
from datetime import datetime



# --- Argument Parsing ---
# Set up an argument parser to accept command-line arguments
parser = argparse.ArgumentParser(description="Run Vidur with different configurations.")
parser.add_argument(
    '--total_gpus', 
    type=int, 
    default=8, 
    help='Total number of GPUs available for the experiment. Default is 8.'
)
parser.add_argument(
    '--log_dir',
    type=str,
    default='./simulator_output',
    help='Base directory to save the log files and outputs. A timestamped subdirectory will be created here. Default is ./logs.'
)
parser.add_argument(
    '--replica_config_device',
    type=str,
    default='a100',
    help='Device type to use for replicas. Default is a100.'
)
parser.add_argument(
    '--network_device',
    type=str,
    default='a100_dgx',
    help='Network device type to use. Default is a100_dgx.'
)
parser.add_argument(
    '--qps',
    type=float,
    default=0,
    help='qps value. default disable with 0 '
)

parser.add_argument(
    '--model',
    type=str,
    default='meta-llama/Meta-Llama-3-8B',
    help='Model name to use. Default is meta-llama/Meta-Llama-3-8B'
)
parser.add_argument(
    '--debug',
    action='store_true',
    help='Enable debug logging'
)
args = parser.parse_args()
TOTAL_GPUS = args.total_gpus
BASE_LOG_DIR = f"{args.log_dir}/compute_{args.replica_config_device}/network_{args.network_device}/{args.model}/qps{int(args.qps) if args.qps > 1 else args.qps}"
NETWORK_DEVICE = args.network_device

# Setup logging based on debug flag
log_level = logging.DEBUG if args.debug else logging.INFO
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- End Argument Parsing ---

# --- Setup ---
# Create a unique, timestamped directory for this run's logs to prevent overwrites.
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
LOG_DIR = os.path.join(BASE_LOG_DIR, timestamp)

# Create the log directory if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)
# --- End Setup ---

# Define the variables and their possible values
cluster_config_num_replicas_list = [1, 2, 4, 8]
replica_config_tensor_parallel_size_list = [1, 2, 4, 8]
replica_config_pipeline_parallel_size_list = [1]

#cluster_config_num_replicas_list = [16]
#replica_config_tensor_parallel_size_list = [1]
#replica_config_pipeline_parallel_size_list = [1, 2, 4, 8]

# Base command template
base_command = (
    "python -m vidur.main "
    "--time_limit 10000 "
    f"--replica_config_model_name {args.model} "
    "--request_generator_config_type synthetic "
    "--synthetic_request_generator_config_num_requests 300 "
    "--length_generator_config_type fixed "
    "--fixed_request_length_generator_config_prefill_tokens 300 "
    "--fixed_request_length_generator_config_decode_tokens 3 "
    #"--interval_generator_config_type static "
    "--interval_generator_config_type poisson "
    f"--poisson_request_interval_generator_config_qps {args.qps} "
    "--global_scheduler_config_type round_robin "
    "--replica_scheduler_config_type vllm_v1 "
    "--vllm_v1_scheduler_config_chunk_size 8192 "
    "--vllm_v1_scheduler_config_batch_size_cap 512 "
    f"--metrics_config_output_dir {BASE_LOG_DIR}"
    #"--cache_config_enable_prefix_caching"
)

logger.info(f"Starting experiment with a total of {TOTAL_GPUS} GPUs available.")
logger.info(f"Logs for this run will be saved in the '{LOG_DIR}' directory.")

# Iterate through all combinations of the variables
for num_replicas in cluster_config_num_replicas_list:
    for tensor_parallel_size in replica_config_tensor_parallel_size_list:
        for pipeline_parallel_size in replica_config_pipeline_parallel_size_list:

            # Calculate the total number of GPUs required for this configuration
            gpus_required = num_replicas * tensor_parallel_size * pipeline_parallel_size

            # Check if the configuration is possible with the available GPUs
            if gpus_required > TOTAL_GPUS:
                logger.warning(f"Skipping impossible configuration: DP={num_replicas}, TP={tensor_parallel_size}, PP={pipeline_parallel_size} (requires {gpus_required} > {TOTAL_GPUS} GPUs)")
                continue  # Skip to the next iteration

            # Create a unique log file name for the current configuration
            log_filename = f"dp{num_replicas}_tp{tensor_parallel_size}_pp{pipeline_parallel_size}.log"
            log_filepath = os.path.join(LOG_DIR, log_filename)

            # Construct the python part of the command
            python_command = (
                f"{base_command} "
                f"--cluster_config_num_replicas {num_replicas} "
                f"--replica_config_tensor_parallel_size {tensor_parallel_size} "
                f"--replica_config_num_pipeline_stages {pipeline_parallel_size} "
                #add network deivice
                f"--replica_config_device {args.replica_config_device} "
                f"--replica_config_network_device {NETWORK_DEVICE}"
            )
            logger.debug(f"Command: {python_command}")

            # Construct the full command to run the python script,
            # and redirect both stdout and stderr to the log file.
            command_to_run = f"{python_command} > {log_filepath} 2>&1"

            logger.info(f"Running DP={num_replicas}, TP={tensor_parallel_size}, PP={pipeline_parallel_size} ({gpus_required} GPUs) -> {log_filepath}")

            try:
                # Execute the command using a shell. We don't need to capture output
                # here anymore since it's being redirected to a file.
                process = subprocess.run(
                    command_to_run, 
                    shell=True,
                    check=True, 
                    text=True,
                    executable='/bin/bash'
                )
                logger.info(f"Configuration completed successfully: {log_filepath}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Configuration failed (exit code {e.returncode}): {log_filepath}")
            except FileNotFoundError:
                logger.error("Shell '/bin/bash' not found. Ensure you're on a Unix-like system.")
                break # Exit the loop if the shell is not found
            except Exception as e:
                logger.error(f"Unexpected error: {e}")


logger.info("All configurations processed.")
