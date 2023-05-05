#!/bin/bash

wandb_dir="../wandb_power_law_dyhpo_bench_pl_abs"
prefix="offline-run-20230430_"

for run_dir in "$wandb_dir"/"$prefix"*; do
  if [ -d "$run_dir" ]; then
    echo "Syncing run directory: $run_dir"
    wandb sync "$run_dir"
  fi
done

