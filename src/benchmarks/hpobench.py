from collections import OrderedDict
from typing import List, Union, Dict, Tuple
from pathlib import Path
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from src.benchmarks.base_benchmark import BaseBenchmark
from ConfigSpace.hyperparameters import FloatHyperparameter, IntegerHyperparameter, Constant, CategoricalHyperparameter, \
    OrdinalHyperparameter

from hpobench.abstract_benchmark import AbstractBenchmark
from hpobench.dependencies.ml.ml_benchmark_template import MLBenchmark
from hpobench.benchmarks.ml.nn_benchmark import NNBenchmark
from hpobench.benchmarks.ml.tabular_benchmark import TabularBenchmark
from hpobench.util.openml_data_manager import get_openmlcc18_taskids


class HPOBench(BaseBenchmark):
    nr_hyperparameters = 1000

    # Declaring the search space for LCBench
    param_space = None
    max_budget = 51
    min_budget = 1
    max_budgets = None
    min_budgets = None
    max_steps = 10

    hp_names = None
    fidelity_names = None
    log_indicator = None
    fidelity_space = None
    fidelity_interval = None
    fidelity_curve_points = None
    fidelity_steps = None

    # if the best value corresponds to a lower value
    minimization_metric = False

    scenario_metric_names = {
        "nn": "function_value",
        "lr": "function_value",
        "svm": "function_value",
        "xgb": "function_value",
        "rf": "function_value",
    }

    scenario_metric_is_minimize = {
        "nn": True,
        "lr": True,
        "svm": True,
        "xgb": True,
        "rf": True,
    }

    scenario_start_index = {
        "nn": 0,
        "lr": 0,
        "svm": 0,
        "xgb": 0,
        "rf": 0,
    }

    is_metric_best_end = {
        'iter': False,
        'subsample': True,
    }

    def __init__(self, path_to_json_files: Path, dataset_name: str, seed=None):
        super().__init__(path_to_json_files, seed=seed)

        # self.scenario_name = "lcbench"
        # self.scenario_name = "taskset"
        self.scenario_name = "nn"
        self.dataset_name = dataset_name
        self.categorical_indicator = None
        self.max_value = 1.0
        self.min_value = 0.0
        self.benchmark_config = None
        self.benchmark_config_pd = None
        self.metric_name = self.scenario_metric_names[self.scenario_name]
        HPOBench.minimization_metric = self.scenario_metric_is_minimize[self.scenario_name]
        HPOBench.min_budget = self.scenario_start_index[self.scenario_name]

        self.benchmark: AbstractBenchmark = self._load_benchmark()
        self.dataset_names = self.load_dataset_names()

    def _load_benchmark(self):
        task_ids = get_openmlcc18_taskids()
        # bench: AbstractBenchmark = TabularBenchmark(task_id=int(self.dataset_name), rng=self.seed)
        bench: AbstractBenchmark = TabularBenchmark(
            model=self.scenario_name, task_id=int(self.dataset_name), rng=self.seed
        )
        # print(bench.instances)

        config_space = bench.get_configuration_space(seed=self.seed)
        HPOBench.param_space = self.extract_hyperparameter_info(config_space=config_space)
        # sort so that categorical columns will be at the end
        HPOBench.param_space = OrderedDict(sorted(HPOBench.param_space.items(), key=lambda item: item[1][2] == 'str'))
        HPOBench.hp_names = list(HPOBench.param_space.keys())

        fidelity_space = bench.get_fidelity_space(seed=self.seed)
        HPOBench.fidelity_space = self.extract_hyperparameter_info(config_space=fidelity_space)
        HPOBench.fidelity_names = list(HPOBench.fidelity_space.keys())

        HPOBench.fidelity_interval = {}
        HPOBench.max_budgets = {}
        HPOBench.min_budgets = {}
        HPOBench.fidelity_curve_points = {}
        HPOBench.fidelity_steps = {}
        for k, v in HPOBench.fidelity_space.items():
            HPOBench.min_budgets[k] = v[0]
            max_budget = v[1]
            if v[2] == 'int':
                interval = 1
                size = v[1] - v[0] + 1
                steps = size
                if size > self.max_steps:
                    max_budget = v[0] + self.max_steps - 1
                    steps = self.max_steps
            elif v[2] == 'float':
                interval = (v[1] - v[0]) / self.max_steps
                steps = self.max_steps
            else:
                raise NotImplementedError
            HPOBench.max_budgets[k] = max_budget
            HPOBench.fidelity_interval[k] = interval
            HPOBench.fidelity_curve_points[k] = np.around(np.arange(v[0], max_budget + interval / 2, interval),
                                                          decimals=4)
            HPOBench.fidelity_steps[k] = steps

        self.categorical_indicator = [v[2] == 'str' for v in HPOBench.param_space.values()]
        self.categories = [v[4] for v in HPOBench.param_space.values() if v[2] == 'str']
        HPOBench.log_indicator = [v[3] for v in HPOBench.param_space.values()]

        self.benchmark_config = self.generate_hyperparameter_candidates(bench)
        self.benchmark_config_pd = pd.DataFrame(self.benchmark_config, columns=HPOBench.hp_names)

        return bench

    def generate_hyperparameter_candidates(self, benchmark) -> List:
        configs = benchmark.get_configuration_space(seed=self.seed).sample_configuration(HPOBench.nr_hyperparameters)

        hp_configs = [config.get_dictionary() for config in configs]

        return hp_configs

    def extract_hyperparameter_info(self, config_space):
        hyperparameter_info = OrderedDict()
        for hp in config_space.get_hyperparameters():
            hp_name = hp.name
            default_value = hp.default_value
            is_log = False
            categories = []
            if isinstance(hp, Constant):
                value = hp.value
                if isinstance(value, float):
                    hp_type = 'float'
                elif isinstance(value, int):
                    hp_type = 'int'
                elif isinstance(value, str):
                    hp_type = 'str'
                    categories = [value]
                else:
                    raise NotImplementedError
                lower = upper = value
            elif isinstance(hp, FloatHyperparameter):
                hp_type = 'float'
                is_log = hp.log
                lower = hp.lower
                upper = hp.upper
            elif isinstance(hp, IntegerHyperparameter):
                hp_type = 'int'
                is_log = hp.log
                lower = hp.lower
                upper = hp.upper
            elif isinstance(hp, CategoricalHyperparameter):
                hp_type = 'str'
                lower = 0
                upper = 0
                categories = hp.choices
            elif isinstance(hp, OrdinalHyperparameter):
                hp_type = 'ord'
                lower = 0
                upper = len(hp.sequence) - 1
                categories = hp.sequence
            else:
                raise NotImplementedError(f"Hyperparameter type not implemented: {hp}")

            hyperparameter_info[hp_name] = [lower, upper, hp_type, is_log, categories, default_value]

        return hyperparameter_info

    def load_dataset_names(self) -> List[str]:
        return get_openmlcc18_taskids()

    def get_hyperparameter_candidates(self) -> pd.DataFrame:
        return self.benchmark_config_pd

    def get_best_performance(self):
        incumbent_curve = self.get_incumbent_curve()
        if self.is_minimize:
            best_value = min(incumbent_curve)
        else:
            best_value = max(incumbent_curve)

        return best_value

    def get_worst_performance(self):
        min_value = np.PINF
        for hp_index in range(0, HPOBench.nr_hyperparameters):
            val_curve, _ = self.get_curve(hp_index=hp_index, budget=HPOBench.max_budgets)
            if self.is_minimize:
                worst_performance_hp_curve = max(val_curve)
            else:
                worst_performance_hp_curve = min(val_curve)
            if worst_performance_hp_curve < min_value:
                min_value = worst_performance_hp_curve

        return min_value

    def get_performance(self, hp_index: int, budget: Union[int, List, Dict]) -> float:
        config_dict = self.benchmark_config[hp_index]

        if isinstance(budget, List):
            fidelity_dict = {k: v for k, v in zip(HPOBench.fidelity_names, budget)}
            config_dict.update(fidelity_dict)
        elif isinstance(budget, int):
            fidelity_dict = {self.fidelity_names[0]: budget}
        else:
            fidelity_dict = budget
        # config_dict.update(fidelity_dict)

        metrics = self.benchmark.objective_function(configuration=config_dict, fidelity=fidelity_dict, seed=self.seed)
        metric = metrics[self.metric_name]

        return metric

    def get_curve(self, hp_index: int, budget: Union[int, Dict]) -> Tuple[List[float], List[Dict]]:
        config_dict = self.benchmark_config[hp_index]

        if isinstance(budget, int):
            budget = {self.fidelity_names[0]: budget}

        valid_curves = []
        for k in self.fidelity_names:
            curve = HPOBench.fidelity_curve_points[k]
            if HPOBench.is_metric_best_end[k]:
                valid_curves.append(curve[curve == budget[k]])
            else:
                valid_curves.append(curve[curve <= budget[k]])

        mesh = np.meshgrid(*valid_curves)

        # Stack the meshgrid to get 2D array of coordinates and reshape it
        fidelity_product = np.dstack(mesh).reshape(-1, len(self.fidelity_names))
        fidelity_dicts = [
            {k: (int(v) if HPOBench.fidelity_space[k][2] == 'int' else v) for k, v in zip(self.fidelity_names, values)}
            for values in fidelity_product]

        # config_dict = [{**config_dict, **dict} for dict in fidelity_dicts]

        metric = []
        for fidelity_dict in fidelity_dicts:
            metrics = self.benchmark.objective_function(
                configuration=config_dict, fidelity=fidelity_dict, seed=self.seed
            )
            metric.append(metrics[self.metric_name])
        # metrics = self.benchmark.objective_function(configuration=config_dict, fidelity=fidelity_dicts, seed=self.seed)
        # metric = [v[self.metric_name] for v in metrics]

        return metric, fidelity_dicts

    def get_curve_best(self, hp_index: int) -> float:
        curve, _ = self.get_curve(hp_index=hp_index, budget=HPOBench.max_budgets)
        if self.is_minimize:
            best_value = min(curve)
        else:
            best_value = max(curve)
        return best_value

    @lru_cache(maxsize=1)
    def get_incumbent_curve(self) -> List[float]:
        best_value = -1
        best_curve = None
        for index in range(0, HPOBench.nr_hyperparameters):
            val_curve, _ = self.get_curve(hp_index=index, budget=HPOBench.max_budgets)
            if self.is_minimize:
                value = min(val_curve)
                if value < best_value:
                    best_value = value
                    best_curve = val_curve
            else:
                value = max(val_curve)
                if value > best_value:
                    best_value = value
                    best_curve = val_curve
        return best_curve

    def get_max_value(self) -> float:
        return max(self.get_incumbent_curve())

    @lru_cache(maxsize=1)
    def get_incumbent_config_id(self) -> int:
        raise NotImplementedError
        best_value = -1
        best_index = -1
        for index in range(0, HPOBench.nr_hyperparameters):
            val_curve, _ = self.get_curve(hp_index=index, budget=HPOBench.max_budgets)
            max_value = min(val_curve)

            if max_value > best_value:
                best_value = max_value
                best_index = index

        return best_index

    def get_gap_performance(self) -> float:

        incumbent_curve = self.get_incumbent_curve()
        if self.is_minimize:
            best_value = min(incumbent_curve)
        else:
            best_value = max(incumbent_curve)
        worst_value = self.get_worst_performance()

        return best_value - worst_value

    def get_step_cost(self, hp_index: int, budget: int):

        time_cost_curve = self.benchmark.query(
            dataset_name=self.dataset_name,
            config_id=hp_index,
            tag='time',
        )
        time_cost_curve = time_cost_curve[1:]
        budget = int(budget)
        if budget > 1:
            step_cost = time_cost_curve[budget - 1] - time_cost_curve[budget - 2]
        else:
            step_cost = time_cost_curve[budget - 1]

        return step_cost

    def set_dataset_name(self, dataset_name: str):
        self.dataset_name = dataset_name
        return self
