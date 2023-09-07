import math
import os
import sys
from collections import OrderedDict
from typing import List
import numpy as np
# from numpy.typing import NDArray
import pandas as pd
from pathlib import Path
import json
import ConfigSpace as CS
from ConfigSpace import Float, Integer, OrdinalHyperparameter, Categorical
from typing import List, Union, Dict, Tuple, Optional, Set
import itertools

import nanogpt_bench

from src.benchmarks.base_benchmark import BaseBenchmark
from src.history.fidelity_manager import FidelityManager


class NanoGPTBench(BaseBenchmark):
    nr_hyperparameters = 12

    param_space = None
    max_budget = 51
    min_budget = 1
    # max_budgets = None
    # min_budgets = None
    max_steps = 10

    hp_names = None
    fidelity_names = None
    log_indicator = None
    fidelity_space = None
    fidelity_interval = None
    # fidelity_curve_points = None
    # fidelity_steps = None

    # if the best value corresponds to a lower value
    minimization_metric = True

    is_metric_best_end = {
        'epochs': False,
        'n_layer': False,
        'n_embd': False,
    }

    def __init__(self, path_to_json_files: Path = "", dataset_name: str = "", seed=None):
        super().__init__(path_to_json_files, seed=seed)
        self.dataset_name = dataset_name
        self.dataset_names = None
        self.categorical_indicator = None
        self.config_ids = {}
        self.benchmark_config = None
        self.benchmark_config_pd = None
        self.fidelity_manager: Optional[FidelityManager] = None
        self.metric_name = 'valid-loss'
        self.all_unique_configurations_added = False
        self.configurations_set: Set[Tuple] = set()

        self.benchmark = self.init_benchmark()

        self.nr_hyperparameters = NanoGPTBench.nr_hyperparameters

    def init_benchmark(self):
        self.minimization_metric = NanoGPTBench.minimization_metric
        task = "openwebtext"
        mode = "table"
        benchmark = nanogpt_bench.Benchmark(
            task=task,
            mode=mode,
            save_dir=self.path_to_json_file,
            download=False if mode == "table" or task == "openwebtext" else True,
            search_space_id=3,
        )

        # config = {
        #     "lr_max": 1e-5,
        #     "lr_min_percent": 1e-2,
        #     "n_embd": 6,
        #     "warmup_percent": 0.2,
        #     "n_layer": 1,
        #     "n_head": 6,
        #     "vocab_size": 50304,
        #     "block_size": 512,
        #     "max_epochs": 350,
        # }
        # # config = {'block_size': 512, 'lr_max': 1e-05, 'lr_min_percent': 0.1, 'max_epochs': 350, 'n_embd': 6,
        # #           'n_head': 6, 'n_layer': 1, 'vocab_size': 50304, 'warmup_percent': 0.2}
        #
        # results = benchmark(
        #     config,
        #     config_id=None,
        #     worker_dir='.',
        #     global_seed=333,
        #     epochs=350,
        #     debug=False,
        #     full_trajectory=False,
        # )

        config_space = self.get_search_space()
        NanoGPTBench.param_space = self.extract_hyperparameter_info(config_space=config_space)
        NanoGPTBench.hp_names = list(NanoGPTBench.param_space.keys())

        raw_fidelity_space = self.get_fidelity_space()
        NanoGPTBench.fidelity_space = self.extract_hyperparameter_info(config_space=raw_fidelity_space)
        NanoGPTBench.fidelity_names = list(NanoGPTBench.fidelity_space.keys())

        NanoGPTBench.fidelity_interval = {}

        self.categorical_indicator = [v[2] == 'str' for v in NanoGPTBench.param_space.values()]
        self.categories = [v[4] for v in NanoGPTBench.param_space.values() if v[2] == 'str']
        NanoGPTBench.log_indicator = [v[3] for v in NanoGPTBench.param_space.values()]

        self.benchmark_config = self.generate_hyperparameter_candidates(config_space)
        self.benchmark_config_pd = pd.DataFrame(self.benchmark_config, columns=NanoGPTBench.hp_names)

        self.fidelity_manager = FidelityManager(
            fidelity_space=raw_fidelity_space,
            num_configurations=NanoGPTBench.nr_hyperparameters,
            max_steps=self.max_steps,
        )

        self.max_budgets = self.fidelity_manager.get_max_fidelity()
        self.min_budgets = self.fidelity_manager.get_min_fidelity()

        return benchmark

    def generate_hyperparameter_candidates(self, benchmark: CS.ConfigurationSpace) -> List[Dict]:
        if self.seed is not None:
            benchmark.seed(seed=self.seed)
        hp_configs = self.sample_configurations(benchmark=benchmark, num_configurations=NanoGPTBench.nr_hyperparameters)
        # configs = benchmark.sample_configuration(NanoGPTBench.nr_hyperparameters)
        # hp_configs = [config.get_dictionary() for config in configs]

        return hp_configs

    def sample_configurations(
        self, benchmark: CS.ConfigurationSpace, num_configurations: int, max_try_limit: int = 100
    ):
        num_added = 0
        num_tries = 0
        hp_configs = []
        while num_added < num_configurations and num_tries < max_try_limit:
            num_configurations_to_fill = num_configurations - num_added
            configurations = benchmark.sample_configuration(num_configurations_to_fill)
            if not isinstance(configurations, List):
                configurations = [configurations]
            configurations = [config.get_dictionary() for config in configurations]
            for config in configurations:
                config_tuple = tuple(sorted(config.items()))
                if config_tuple not in self.configurations_set or self.all_unique_configurations_added:
                    hp_configs.append(config)
                    self.configurations_set.add(config_tuple)
                    num_added += 1
            num_tries += 1

        if num_tries == max_try_limit:
            self.all_unique_configurations_added = True
            next_hp_configs = self.sample_configurations(
                benchmark=benchmark, num_configurations=num_configurations - num_added
            )
            hp_configs.extend(next_hp_configs)

        return hp_configs

    def get_hyperparameter_candidates(self) -> np.ndarray:
        return self.benchmark_config_pd

    def get_benchmark_config(self, config_id):
        column_name = self.config_ids[config_id]
        config = {}
        for i, hp_name in enumerate(self.hp_names):
            config[hp_name] = column_name[i]
        return config

    def get_curve(self, hp_index: int, budget: Union[int, Dict]) -> Tuple[List[float], List[Dict]]:
        # config_dict = self.benchmark_config[hp_index]

        if isinstance(budget, int):
            budget = {self.fidelity_names[0]: budget}

        valid_curves = []
        for k in self.fidelity_names:
            curve = self.fidelity_manager.fidelity_space[k]
            # if k in NanoGPTBench.is_metric_best_end and NanoGPTBench.is_metric_best_end[k]:
            #     valid_curves.append(curve[curve == budget[k]])
            # else:
            valid_curves.append(curve[curve <= budget[k]])

        fidelity_product = list(itertools.product(*valid_curves))
        fidelity_dicts = [
            {k: (int(v) if NanoGPTBench.fidelity_space[k][2] == 'int' else v)
             for k, v in zip(self.fidelity_names, values)}
            for values in fidelity_product]

        # config_dict = [{**config_dict, **dict} for dict in fidelity_dicts]

        metrics = []
        for fidelity_dict in fidelity_dicts:
            metric, _ = self.get_performance(hp_index=hp_index, fidelity_id=fidelity_dict)
            metrics.append(metric)
        # metrics = self.benchmark.objective_function(configuration=config_dict, fidelity=fidelity_dicts, seed=self.seed)
        # metric = [v[self.metric_name] for v in metrics]

        return metrics, fidelity_dicts

    def get_performance(self, hp_index: int, fidelity_id: Union[Tuple[int], Dict]) -> Tuple[float, float]:
        config_dict = self.benchmark_config[hp_index]

        # if isinstance(budget, List):
        #     fidelity_dict = {k: v for k, v in zip(NanoGPTBench.fidelity_names, budget)}
        #     config_dict.update(fidelity_dict)
        # elif isinstance(budget, int):
        #     fidelity_dict = {self.fidelity_names[0]: budget}
        # else:
        #     fidelity_dict = budget
        # config_dict.update(fidelity_dict)
        # fidelity_dict = self.fidelity_manager.get_fidelities(fidelity_id, return_dict=True)
        if not isinstance(fidelity_id, Dict):
            fidelity_dict = dict(zip(self.fidelity_names, fidelity_id))
        else:
            fidelity_dict = fidelity_id

        epoch = fidelity_dict["epochs"]
        config_dict = {**config_dict, **{k: v for k, v in fidelity_dict.items() if k != 'epochs'}}

        try:
            metric_data = self.benchmark(
                config_dict,
                global_seed=333,
                epochs=epoch,
                debug=False,
                full_trajectory=False,
            )
            metric = metric_data[self.metric_name]
            eval_time = metric_data['runtime']
        except KeyError as ex:
            # metric = np.nan
            # Simulate failed runs.
            try:
                metric_data = self.benchmark(
                    config_dict,
                    global_seed=333,
                    epochs=epoch,
                    debug=False,
                    full_trajectory=True,
                )
                eval_time = metric_data[-1]['runtime']
                metric = np.nan
            except KeyError as ex_in:
                eval_time = 0
                metric = np.nan

        if np.isnan(metric).any():
            a = 0

        return metric, eval_time

    def get_fidelity_manager(self):
        return self.fidelity_manager

    def set_dataset_name(self, dataset_name):
        self.dataset_name = dataset_name
        return self

    def get_search_space(self):
        config_space = CS.ConfigurationSpace("nanogpt_bench")
        config_space.add_hyperparameter(
            CS.Constant(
                "max_epochs",
                value=350,
                # default_value=350,
                # meta="Maximum number of epochs (default: 350)"
            )
        )
        config_space.add_hyperparameter(
            CS.Constant(
                "n_head",
                value=6,
                # default_value=6,
                # meta="Number of attention heads and layers (default: 6)"
            )
        )
        # The vocabulary size is fixed to 65 for the Shakespeare character-level dataset.
        # For other datasets, it is equal to GPT2 tokenization size.
        # Kept here for convenience, will be overridden in the benchmark (train_config)
        # based on dataset meta-information.
        config_space.add_hyperparameter(
            CS.Constant(
                "vocab_size",
                value=50304,
                # default_value=50304,
                # meta="Vocabulary size (default: 50304)"
            )
        )
        config_space.add_hyperparameter(
            CS.Constant(
                "block_size",
                value=512,
                # default_value=512,
                # meta="Context block size (default: 512)"
            )
        )
        config_space.add_hyperparameter(
            CS.OrdinalHyperparameter(
                "lr_max",
                sequence=[1e-5, 1e-4, 1e-3],
                # default_value=1e-4,
                # meta="Maximum learning rate (default: 1e-4)"
            )
        )
        config_space.add_hyperparameter(
            CS.OrdinalHyperparameter(
                "lr_min_percent",
                sequence=[1e-2, 1e-1],
                # default_value=1e-1,
                # meta="Learning rate lower bound: factor to multiply the lr_max by (default: 1e-1)"
            )
        )
        config_space.add_hyperparameter(
            CS.OrdinalHyperparameter(
                "warmup_percent",
                sequence=[0.10, 0.2],
                # default_value=0.05,
                # meta="Factor to multiply the number of epochs by to determine the warmup epochs (default: 0.05)"
            )
        )

        return config_space

    def get_fidelity_space(self):
        config_space = CS.ConfigurationSpace("nanogpt_bench_fidelity")
        config_space.add_hyperparameter(
            CS.OrdinalHyperparameter(
                "epochs",
                # sequence=list(range(1, 350)),
                sequence=[350],
            )
        )
        config_space.add_hyperparameter(
            CS.OrdinalHyperparameter(
                "n_embd",
                sequence=[6, 12, 24, 48, 96, 192, 384],
                # default_value=6,
                # meta="Embedding dimension (default: 6)"
            )
        )
        config_space.add_hyperparameter(
            CS.OrdinalHyperparameter(
                "n_layer",
                sequence=[1, 2, 3, 4, 5, 6],
            )
        )
        return config_space
