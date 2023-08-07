#!/bin/bash

export ROOT_PATH=/home/fr/fr_fr/fr_rs442/project/DPL
export JOB_PATH=/home/fr/fr_fr/fr_rs442/project/DPL/nemo_jobs
export ENV_PATH=/home/fr/fr_fr/fr_rs442/.conda/envs/modpl

export use_config=true
#"$ROOT_PATH"/bash_scripts/experiment_run_array.sh "taskset" "power_law" "$ROOT_PATH/results" "$use_config"
#"$ROOT_PATH"/bash_scripts/experiment_run_array.sh "lcbench" "power_law" "$ROOT_PATH/results" "$use_config"
#"$ROOT_PATH"/bash_scripts/experiment_run_array.sh "lcbench_mini" "power_law" "$ROOT_PATH/results" "$use_config"

"$ROOT_PATH"/bash_scripts/experiment_run_array.sh "synthetic_mf" "random" "$ROOT_PATH/results" "$use_config"

#"$ROOT_PATH"/bash_scripts/experiment_run_array.sh "taskset" "dyhpo" "$ROOT_PATH/results" "$use_config"
#"$ROOT_PATH"/bash_scripts/experiment_run_array.sh "lcbench" "dyhpo" "$ROOT_PATH/results" "$use_config"
#"$ROOT_PATH"/bash_scripts/experiment_run_array.sh "lcbench_mini" "dyhpo" "$ROOT_PATH/results" "$use_config"

#"$ROOT_PATH"/bash_scripts/experiment_run_array.sh "yahpo" "power_law" "$ROOT_PATH/results" "$use_config"
#"$ROOT_PATH"/bash_scripts/experiment_run_array.sh "yahpo" "dyhpo" "$ROOT_PATH/results" "$use_config"

#"$ROOT_PATH"/bash_scripts/experiment_run_array.sh "taskset" "random" "$ROOT_PATH/results"
#"$ROOT_PATH"/bash_scripts/experiment_run_array.sh "taskset" "asha" "$ROOT_PATH/results"
#"$ROOT_PATH"/bash_scripts/experiment_run_array.sh "taskset" "dehb" "$ROOT_PATH/results"
#"$ROOT_PATH"/bash_scripts/experiment_run_array.sh "taskset" "dragonfly" "$ROOT_PATH/results"

#"$ROOT_PATH"/bash_scripts/experiment_run_array.sh "lcbench" "random" "$ROOT_PATH/results"
#"$ROOT_PATH"/bash_scripts/experiment_run_array.sh "lcbench" "asha" "$ROOT_PATH/results"
#"$ROOT_PATH"/bash_scripts/experiment_run_array.sh "lcbench" "dehb" "$ROOT_PATH/results"
#"$ROOT_PATH"/bash_scripts/experiment_run_array.sh "lcbench" "dragonfly" "$ROOT_PATH/results"

#"$ROOT_PATH"/bash_scripts/experiment_run_array.sh "lcbench_mini" "random" "$ROOT_PATH/results"
#"$ROOT_PATH"/bash_scripts/experiment_run_array.sh "lcbench_mini" "asha" "$ROOT_PATH/results"
#"$ROOT_PATH"/bash_scripts/experiment_run_array.sh "lcbench_mini" "dehb" "$ROOT_PATH/results"
#"$ROOT_PATH"/bash_scripts/experiment_run_array.sh "lcbench_mini" "dragonfly" "$ROOT_PATH/results"
