from collections import OrderedDict
from typing import List, Union, Dict, Tuple
from pathlib import Path

import numpy as np
# from numpy.typing import NDArray
import pandas as pd

from src.benchmarks.base_benchmark import BaseBenchmark
from yahpo_gym import benchmark_set, local_config, list_scenarios, BenchmarkSet
from ConfigSpace.hyperparameters import FloatHyperparameter, IntegerHyperparameter, Constant, CategoricalHyperparameter


class YAHPOGym(BaseBenchmark):
    nr_hyperparameters = 2000

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
        "lcbench": "val_balanced_accuracy",
        "taskset": "val1",
        "rbv2_svm": "acc",
    }

    scenario_metric_is_minimize = {
        "lcbench": False,
        "taskset": True,
        "rbv2_svm": False,
    }

    scenario_start_index = {
        "lcbench": 1,
        "taskset": 0,
        "rbv2_svm": 0,
    }

    def __init__(self, path_to_json_files: Path, dataset_name: str, seed=None):
        super().__init__(path_to_json_files, seed=seed)

        local_config.init_config()
        local_config.set_data_path(path_to_json_files)

        # self.scenario_name = "lcbench"
        # self.scenario_name = "taskset"
        self.scenario_name = "rbv2_svm"
        self.dataset_name = dataset_name
        self.categorical_indicator = None
        self.max_value = 1.0
        self.min_value = 0.0
        self.benchmark_config = None
        self.benchmark_config_pd = None
        self.metric_name = self.scenario_metric_names[self.scenario_name]
        YAHPOGym.minimization_metric = self.scenario_metric_is_minimize[self.scenario_name]
        YAHPOGym.min_budget = self.scenario_start_index[self.scenario_name]

        self.benchmark: BenchmarkSet = self._load_benchmark()
        self.dataset_names = self.load_dataset_names()

    def _load_benchmark(self):
        bench: BenchmarkSet = benchmark_set.BenchmarkSet(scenario=self.scenario_name)
        # print(bench.instances)
        bench.set_instance(self.dataset_name)

        config_space = bench.get_opt_space(drop_fidelity_params=True, seed=self.seed)
        YAHPOGym.param_space = self.extract_hyperparameter_info(config_space=config_space)
        # sort so that categorical columns will be at the end
        YAHPOGym.param_space = OrderedDict(sorted(YAHPOGym.param_space.items(), key=lambda item: item[1][2] == 'str'))
        YAHPOGym.hp_names = list(YAHPOGym.param_space.keys())

        fidelity_space = bench.get_fidelity_space(seed=self.seed)
        YAHPOGym.fidelity_space = self.extract_hyperparameter_info(config_space=fidelity_space)
        YAHPOGym.fidelity_names = list(YAHPOGym.fidelity_space.keys())

        YAHPOGym.fidelity_interval = {}
        YAHPOGym.max_budgets = {}
        YAHPOGym.min_budgets = {}
        YAHPOGym.fidelity_curve_points = {}
        YAHPOGym.fidelity_steps = {}
        for k, v in YAHPOGym.fidelity_space.items():
            YAHPOGym.min_budgets[k] = v[0]
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
            YAHPOGym.max_budgets[k] = max_budget
            YAHPOGym.fidelity_interval[k] = interval
            YAHPOGym.fidelity_curve_points[k] = np.around(np.arange(v[0], max_budget + interval, interval), decimals=4)
            YAHPOGym.fidelity_steps[k] = steps

        self.categorical_indicator = [v[2] == 'str' for v in YAHPOGym.param_space.values()]
        self.categories = [v[4] for v in YAHPOGym.param_space.values()]
        YAHPOGym.log_indicator = [v[3] for v in YAHPOGym.param_space.values()]

        self.benchmark_config = self.generate_hyperparameter_candidates(bench)
        self.benchmark_config_pd = pd.DataFrame(self.benchmark_config, columns=YAHPOGym.hp_names)

        return bench

    def generate_hyperparameter_candidates(self, benchmark) -> List:
        configs = benchmark.get_opt_space(drop_fidelity_params=True, seed=self.seed).sample_configuration(
            YAHPOGym.nr_hyperparameters)

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
            else:
                raise NotImplementedError(f"Hyperparameter type not implemented: {hp}")

            hyperparameter_info[hp_name] = [lower, upper, hp_type, is_log, categories, default_value]

        return hyperparameter_info

    def load_dataset_names(self) -> List[str]:
        return self.benchmark.instances

    def get_hyperparameter_candidates(self) -> pd.DataFrame:
        return self.benchmark_config_pd

    def get_best_performance(self):
        incumbent_curve = self.get_incumbent_curve()
        best_value = max(incumbent_curve)

        return best_value

    def get_worst_performance(self):
        min_value = np.PINF
        for hp_index in range(0, YAHPOGym.nr_hyperparameters):
            val_curve, _ = self.get_curve(hp_index=hp_index, budget=YAHPOGym.max_budgets)
            worst_performance_hp_curve = min(val_curve)
            if worst_performance_hp_curve < min_value:
                min_value = worst_performance_hp_curve

        return min_value

    def get_performance(self, hp_index: int, budget: Union[int, List, Dict]) -> float:
        config_dict = self.benchmark_config[hp_index]

        if isinstance(budget, List):
            fidelity_dict = {k: v for k, v in zip(YAHPOGym.fidelity_names, budget)}
            config_dict.update(fidelity_dict)
        elif isinstance(budget, int):
            fidelity_dict = {self.fidelity_names[0]: budget}
        else:
            fidelity_dict = budget
        config_dict.update(fidelity_dict)

        metrics = self.benchmark.objective_function(config_dict, seed=self.seed)
        metric = metrics[0][self.metric_name]

        return metric

    def get_curve(self, hp_index: int, budget: Union[int, Dict]) -> Tuple[List[float], List[Dict]]:
        config_dict = self.benchmark_config[hp_index]

        if isinstance(budget, int):
            budget = {self.fidelity_names[0]: budget}

        valid_curves = []
        for k in self.fidelity_names:
            curve = YAHPOGym.fidelity_curve_points[k]
            valid_curves.append(curve[curve <= budget[k]])

        mesh = np.meshgrid(*valid_curves)

        # Stack the meshgrid to get 2D array of coordinates and reshape it
        fidelity_product = np.dstack(mesh).reshape(-1, len(self.fidelity_names))
        fidelity_dicts = [
            {k: (int(v) if YAHPOGym.fidelity_space[k][2] == 'int' else v) for k, v in zip(self.fidelity_names, values)}
            for values in fidelity_product]

        config_dict = [{**config_dict, **dict} for dict in fidelity_dicts]

        metrics = self.benchmark.objective_function(config_dict, seed=self.seed)
        metric = [v[self.metric_name] for v in metrics]

        return metric, fidelity_dicts

    def get_curve_best(self, hp_index: int) -> float:
        curve, _ = self.get_curve(hp_index=hp_index, budget=YAHPOGym.max_budgets)
        best_value = max(curve)
        return best_value

    def get_incumbent_curve(self) -> List[float]:
        best_value = -1
        best_curve = None
        for index in range(0, YAHPOGym.nr_hyperparameters):
            val_curve, _ = self.get_curve(hp_index=index, budget=YAHPOGym.max_budgets)
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
        for index in range(0, YAHPOGym.nr_hyperparameters):
            val_curve, _ = self.get_curve(hp_index=index, budget=YAHPOGym.max_budgets)
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
