#!/bin/bash

export ROOT_PATH=/home/fr/fr_fr/fr_rs442/project/DPL
export JOB_PATH=/home/fr/fr_fr/fr_rs442/project/DPL/nemo_jobs
export ENV_PATH=/home/fr/fr_fr/fr_rs442/.conda/envs/modpl

# "$ROOT_PATH"/bash_scripts/experiment_run_array.sh "taskset" "power_law" ""$ROOT_PATH"/results"
 "$ROOT_PATH"/bash_scripts/experiment_run_array.sh "lcbench_mini" "dyhpo" "$ROOT_PATH"/results


# "$ROOT_PATH"/bash_scripts/experiment_run_array.sh "taskset" "random" "$ROOT_PATH/results"
# "$ROOT_PATH"/bash_scripts/experiment_run_array.sh "taskset" "asha" "$ROOT_PATH/results"
# "$ROOT_PATH"/bash_scripts/experiment_run_array.sh "taskset" "dehb" "$ROOT_PATH/results"
# "$ROOT_PATH"/bash_scripts/experiment_run_array.sh "taskset" "dragonfly" ""$ROOT_PATH"/results"

# "$ROOT_PATH"/bash_scripts/experiment_run_array.sh "lcbench" "random" ""$ROOT_PATH"/results"
# "$ROOT_PATH"/bash_scripts/experiment_run_array.sh "lcbench" "asha" ""$ROOT_PATH"/results"
# "$ROOT_PATH"/bash_scripts/experiment_run_array.sh "lcbench" "dehb" ""$ROOT_PATH"/results"
# "$ROOT_PATH"/bash_scripts/experiment_run_array.sh "lcbench" "dragonfly" ""$ROOT_PATH"/results"
