#!/bin/bash

unique_id=$(date +"%Y%m%d_%H%M%S")
job_directory="$JOB_PATH/job_$unique_id"
mkdir -p "$job_directory"

export JOB_DIRECTORY="$job_directory"
echo "$JOB_DIRECTORY"

cp -R "$ROOT_PATH"/src "$job_directory/"
cp -R "$ROOT_PATH"/*.py "$job_directory/"
cp -R "$ROOT_PATH"/configurations "$job_directory/"
ln -s "$ROOT_PATH"/lc_bench "$job_directory/"
ln -s "$ROOT_PATH"/data "$job_directory/"
ln -s "$ROOT_PATH"/cached "$job_directory/"

# stores WANDB_API_KEY 
secrets_file="$ROOT_PATH/bash_scripts/secrets.sh"
if [ -e "$secrets_file" ]; then
	chmod +x "$secrets_file"
    . "$secrets_file"        # execute the secrets.sh file
fi

export benchmark=$1

if [ "$benchmark" == "lcbench" ]
then
  file="$ROOT_PATH/bash_scripts/lcbench_dataset_names.txt"
elif [ "$benchmark" == "lcbench_mini" ]
then
  file="$ROOT_PATH/bash_scripts/lcbench_mini_dataset_names.txt"
elif [ "$benchmark" == "taskset" ]
then
  file="$ROOT_PATH/bash_scripts/taskset_dataset_names.txt"
elif [ "$benchmark" == "nasbench201" ]
then
  file="$ROOT_PATH/bash_scripts/nasbench_dataset_names.txt"
else
  file="$ROOT_PATH/bash_scripts/pd1_dataset_names.txt"
fi

export surrogate="$2"
export dir="$3"
export use_config="$4"

if [ "$use_config" == true ]
then
  if [ "$surrogate" == "power_law" ]
  then
    config_file="$ROOT_PATH/configurations/power_law_configuration.json"
  elif [ "$surrogate" == "dyhpo" ]
  then
    config_file="$ROOT_PATH/configurations/dyhpo_configuration.json"
  fi
fi

export config="$config_file"

if ! [ -e "$file" ] ; then     # spaces inside square brackets
    echo "$0: $file does not exist" >&2  # error message includes $0 and goes to stderr
    exit 1                   # exit code is non-zero for error
fi

NAMES=$(<$file)
for NAME in $NAMES
do
   export dataset=$(echo $NAME)
   msub -V -t 1-10 "$ROOT_PATH"/bash_scripts/experiment_array.moab
  # "$ROOT_PATH"/bash_scripts/experiment_array.moab
done
