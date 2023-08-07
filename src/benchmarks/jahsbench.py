from collections import OrderedDict
from typing import List, Union, Dict
from pathlib import Path

import ConfigSpace as CS
import numpy as np
# from numpy.typing import NDArray
import pandas as pd

import jahs_bench
from jahs_bench import Benchmark
from jahs_bench.lib.core.configspace import joint_config_space

from src.benchmarks.base_benchmark import BaseBenchmark
from yahpo_gym import benchmark_set, local_config, list_scenarios, BenchmarkSet
from ConfigSpace.hyperparameters import FloatHyperparameter, IntegerHyperparameter, Constant, \
    CategoricalHyperparameter, OrdinalHyperparameter


class JAHSBench(BaseBenchmark):
    nr_hyperparameters = 100

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
        "jahsbench": "val_balanced_accuracy",
    }

    scenario_metric_is_minimize = {
        "jahsbench": False,
    }

    scenario_start_index = {
        "jahsbench": 0,
    }

    def __init__(self, path_to_json_files: Path, dataset_name: str, seed=None):
        super().__init__(path_to_json_files, seed=seed)

        self.scenario_name = "jahsbench"
        self.dataset_name = dataset_name
        self.categorical_indicator = None
        self.max_value = 1.0
        self.min_value = 0.0
        self.benchmark_config = None
        self.benchmark_config_pd = None
        self.metric_name = self.scenario_metric_names[self.scenario_name]
        JAHSBench.minimization_metric = self.scenario_metric_is_minimize[self.scenario_name]
        JAHSBench.min_budget = self.scenario_start_index[self.scenario_name]

        self.benchmark: Benchmark = self._load_benchmark()
        self.dataset_names = self.load_dataset_names()

    def _load_benchmark(self):
        bench: Benchmark = jahs_bench.Benchmark(
            task=self.dataset_name, metrics=["valid-acc"], download=False, lazy=True
        )
        # print(bench.instances)
        print("loaded benchmark")
        filderlity_names = ['N', 'W', 'Resolution']

        config_space = CS.ConfigurationSpace()
        for hp in joint_config_space.get_hyperparameters():
            if hp.name not in filderlity_names:
                config_space.add_hyperparameter(joint_config_space.get_hyperparameter(hp.name))

        JAHSBench.param_space = self.extract_hyperparameter_info(config_space=config_space)
        # sort so that categorical columns will be at the end
        JAHSBench.param_space = OrderedDict(sorted(JAHSBench.param_space.items(), key=lambda item: item[1][2] == 'str'))
        JAHSBench.hp_names = list(JAHSBench.param_space.keys())

        fidelity_space = CS.ConfigurationSpace()
        for hp in joint_config_space.get_hyperparameters():
            if hp.name in filderlity_names:
                fidelity_space.add_hyperparameter(joint_config_space.get_hyperparameter(hp.name))
        fidelity_space.add_hyperparameter(CS.UniformIntegerHyperparameter('epoch', lower=1, upper=200, log=False))
        JAHSBench.fidelity_space = self.extract_hyperparameter_info(config_space=fidelity_space)
        JAHSBench.fidelity_names = list(JAHSBench.fidelity_space.keys())

        JAHSBench.fidelity_interval = {}
        JAHSBench.max_budgets = {}
        JAHSBench.min_budgets = {}
        JAHSBench.fidelity_curve_points = {}
        # for k, v in JAHSBench.fidelity_space.items():
        #     JAHSBench.min_budgets[k] = v[0]
        #     max_budget = v[1]
        #     if v[2] == 'int':
        #         interval = 1
        #         size = v[1] - v[0] + 1
        #         steps = size
        #         if size > self.max_budget:
        #             max_budget = v[0] + self.max_budget - 1
        #             steps = self.max_steps
        #     elif v[2] == 'float':
        #         interval = (v[1] - v[0]) / self.max_budget
        #         steps = self.max_steps
        #     elif v[2] == 'ord':
        #         interval = (v[1] - v[0]) / self.max_budget
        #         steps = self.max_steps
        #     else:
        #         raise NotImplementedError
        #     JAHSBench.max_budgets[k] = max_budget
        #     JAHSBench.fidelity_interval[k] = interval
        #     JAHSBench.fidelity_curve_points[k] = np.around(np.arange(v[0], max_budget + interval / 2, interval),
        #                                                    decimals=4)
        #     JAHSBench.fidelity_steps[k] = steps

        self.categorical_indicator = [v[2] == 'str' for v in JAHSBench.param_space.values()]
        self.categories = [v[4] for v in JAHSBench.param_space.values() if v[2] == 'str']
        JAHSBench.log_indicator = [v[3] for v in JAHSBench.param_space.values()]

        self.benchmark_config = self.generate_hyperparameter_candidates(bench)
        self.benchmark_config_pd = pd.DataFrame(self.benchmark_config, columns=JAHSBench.hp_names)

        return bench

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
        return self.benchmark.instances

    def generate_hyperparameter_candidates(self, benchmark) -> np.ndarray:
        hp_configs = [benchmark.sample_config() for _ in range(JAHSBench.nr_hyperparameters)]

        return hp_configs

    def get_hyperparameter_candidates(self) -> pd.DataFrame:
        return self.benchmark_config_pd

    def get_best_performance(self):
        incumbent_curve = self.get_incumbent_curve()
        best_value = max(incumbent_curve)

        return best_value

    def get_worst_performance(self):
        min_value = np.PINF
        for hp_index in range(0, JAHSBench.nr_hyperparameters):
            val_curve = self.get_curve(hp_index=hp_index, budget=JAHSBench.max_budgets)
            worst_performance_hp_curve = min(val_curve)
            if worst_performance_hp_curve < min_value:
                min_value = worst_performance_hp_curve

        return min_value

    def get_performance(self, hp_index: int, budget: Union[int, List]) -> float:
        config_dict = self.benchmark_config[hp_index]

        if isinstance(budget, List):
            fidelity_dict = {k: v for k, v in zip(JAHSBench.fidelity_names, budget)}
            config_dict.update(fidelity_dict)
        elif isinstance(budget, Dict):
            fidelity_dict = {self.fidelity_names[0]: budget}
        else:
            fidelity_dict = budget
        config_dict.update(fidelity_dict)

        metrics = self.benchmark(config_dict, nepochs=200)
        metric = metrics[self.metric_name]

        return metric

    def get_curve(self, hp_index: int, budget: Union[int, Dict], prev_budget: Union[int, Dict] = None) -> List[float]:
        config_dict = self.benchmark_config[hp_index]

        if isinstance(budget, int):
            budget = {self.fidelity_names[0]: budget}

        valid_curves = []
        for k in self.fidelity_names:
            curve = JAHSBench.fidelity_curve_points[k]
            if JAHSBench.is_metric_best_end[k]:
                valid_curves.append(curve[curve == budget[k]])
            else:
                valid_curves.append(curve[curve <= budget[k]])

        mesh = np.meshgrid(*valid_curves)

        # Stack the meshgrid to get 2D array of coordinates and reshape it
        fidelity_product = np.dstack(mesh).reshape(-1, len(self.fidelity_names))
        fidelity_dicts = [
            {k: (int(v) if JAHSBench.fidelity_space[k][2] == 'int' else v) for k, v in zip(self.fidelity_names, values)}
            for values in fidelity_product]

        config_dict = [{**config_dict, **dict} for dict in fidelity_dicts]

        metrics = self.benchmark(config_dict, nepochs=200)
        metric = [v[self.metric_name] for v in metrics]

        return metric

    def get_curve_best(self, hp_index: int) -> float:
        curve = self.get_curve(hp_index=hp_index, budget=JAHSBench.max_budgets)
        best_value = max(curve)
        return best_value

    def get_incumbent_curve(self) -> List[float]:
        best_value = -1
        best_curve = None
        for index in range(0, JAHSBench.nr_hyperparameters):
            val_curve = self.get_curve(hp_index=index, budget=JAHSBench.max_budgets)
            max_value = max(val_curve)
            if max_value > best_value:
                best_value = max_value
                best_curve = val_curve
        return best_curve

    def get_max_value(self) -> float:
        return max(self.get_incumbent_curve())

    def get_incumbent_config_id(self) -> int:
        best_value = -1
        best_index = -1
        for index in range(0, JAHSBench.nr_hyperparameters):
            val_curve = self.get_curve(hp_index=index, budget=JAHSBench.max_budgets)
            max_value = max(val_curve)

            if max_value > best_value:
                best_value = max_value
                best_index = index

        return best_index

    def get_gap_performance(self) -> float:

        incumbent_curve = self.get_incumbent_curve()
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
