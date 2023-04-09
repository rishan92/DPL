from pathlib import Path
import json
import seaborn as sns
import pandas as pd
import pytest
import re
import itertools
import sys
import time

from main_experiment import main

IS_DEV = 0


def test_result_consistancy():
    if not IS_DEV:
        golden_result_path = Path("../Golden_Results")
        result_path = Path("../results")
        result_path = Path("../test_output")
        budget_limit = 1000
    else:
        golden_result_path = Path("../Golden_Test_Results")
        result_path = Path("../test_output")
        budget_limit = 100
    dataset_files_path = Path("../bash_scripts")
    seed_count = 10

    method_names = [
        'random',
        'power_law',
        'asha',
        'dehb',
        # 'dragonfly',
    ]

    benchmark_list = {
        # "taskset": "taskset_dataset_names.txt",
        "lcbench": "lcbench_dataset_names.txt",
    }

    # benchmark_name = "lcbench"
    # dataset_name = "APSFailure"
    checked_file_count = 0

    for benchmark_name, dataset_file_name in benchmark_list.items():

        with open(dataset_files_path / dataset_file_name) as f:
            data = f.readlines()
        delimiters = " ", "\n"
        regex_pattern = '|'.join(map(re.escape, delimiters))
        datasets = re.split(regex_pattern, data[0])

        files_to_check_params = itertools.product(method_names, datasets, range(seed_count))

        for method_name, dataset_name, repeat_nr in files_to_check_params:
            result_file = result_path / benchmark_name / method_name / f'{dataset_name}_{repeat_nr}.json'

            if not result_file.exists():
                print(f"Consistancy Warning: Result File Does Not Exist {benchmark_name=} {method_name=} "
                      f"{dataset_name=} {repeat_nr=}")
                continue

            with open(result_file, 'r') as fp:
                result_info = json.load(fp)

            golden_result_file = golden_result_path / benchmark_name / method_name / f'{dataset_name}_{repeat_nr}.json'
            with open(golden_result_file, 'r') as fp:
                golden_result_info = json.load(fp)

            checked_file_count += 1
            for tag, result in result_info.items():
                if tag == 'overhead':
                    continue
                golden_result = golden_result_info[tag]
                if len(result) != len(golden_result):
                    print(f"\nConsistancy Warning: Result Array Mismatch {benchmark_name=} {method_name=} "
                          f"{dataset_name=} {repeat_nr=} {tag=} result size={len(result)} "
                          f"golden result size={len(golden_result)}\n")
                assert len(result) <= len(golden_result), \
                    f"Consistancy Failed: Result Array Larger Than Golden {benchmark_name=} {method_name=} " \
                    f"{dataset_name=} {repeat_nr=} {tag=} result size={len(result)} " \
                    f"golden result size={len(golden_result)}"
                for i, v in enumerate(result):
                    assert v == golden_result[i], \
                        f"Consistancy Failed: Result Value Mismatch {benchmark_name=} {method_name=} {dataset_name=} " \
                        f"{repeat_nr=} {tag=} epoch={v}"

    assert checked_file_count > 0, f"Consistancy Failed: No Files To Test"
