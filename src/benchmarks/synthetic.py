import math
from collections import OrderedDict
from typing import List
import numpy as np
# from numpy.typing import NDArray
import pandas as pd
from pathlib import Path

from src.benchmarks.base_benchmark import BaseBenchmark


class SyntheticBench(BaseBenchmark):
    nr_hyperparameters = 1

    # Declaring the search space for LCBench
    param_space = OrderedDict([
        ('h1', [0, 1, int, False]),
    ])
    max_budget = 51
    min_budget = 1

    hp_names = list(param_space.keys())

    log_indicator = [False]

    # if the best value corresponds to a lower value
    minimization_metric = True

    def __init__(self, path_to_json_files: Path = "", dataset_name: str = ""):
        super().__init__(path_to_json_files)
        self.dataset_name = dataset_name
        self.dataset_names = None
        self.categorical_indicator = [False] * len(self.param_space)
        self.max_value = 1.0
        self.min_value = 0.0
        self.benchmark_data = None
        self.config_ids = None
        self.init_benchmark()

    def get_synthetic_curve(self, y1, y2, alphas, max_budget):
        betas = y2 - alphas
        gammas = math.log((y2 - alphas) / (y1 - alphas)) / math.log(1 / max_budget)

        scaled_budgets = np.arange(1, max_budget + 1) / self.max_budget
        curve = alphas + betas * np.power(scaled_budgets, -1 * gammas)
        return curve

    def init_benchmark(self):
        iterables = [[h for h in range(self.param_space['h1'][0], self.param_space['h1'][1])]]
        self.config_ids = {0: [0]}
        column_index = pd.MultiIndex.from_product(iterables, names=self.hp_names)
        self.benchmark_data = pd.DataFrame(index=np.arange(1, self.max_budget + 1), columns=column_index)

        if self.dataset_name == "imperfect":
            curve = np.zeros((self.max_budget,))
            curve[:5] = -0.06 * np.arange(0, 5) + 0.8
            curve[5:] = 0.5
        elif self.dataset_name == "concave":
            curve = np.zeros((self.max_budget,))
            split = self.max_budget // 2
            curve1 = self.get_synthetic_curve(y1=0.8, y2=0.3, alphas=0.15, max_budget=split + 1)

            curve[:split + 1] = curve1
            curve[split:] = curve1[::-1]
        elif self.dataset_name == "sin":
            curve = np.zeros((self.max_budget,))
            angle = np.arange(0, self.max_budget) / 7
            curve[:] = (np.sin(angle) + 1) * 0.25 + 0.25
        else:
            if self.dataset_name == "perfect":
                y1 = 0.8
                y2 = 0.3
                alphas = 0.15
            elif self.dataset_name == "thin":
                y1 = 0.42
                y2 = 0.37
                alphas = 0.36
            elif self.dataset_name == "upper":
                y1 = 0.3
                y2 = 0.8
                alphas = 0.9
            else:
                raise NotImplementedError

            curve = self.get_synthetic_curve(y1=y1, y2=y2, alphas=alphas, max_budget=self.max_budget)

        self.benchmark_data.loc[:, 0] = curve

    def get_hyperparameter_candidates(self) -> np.ndarray:

        hp_names = list(self.param_space.keys())
        hp_configs = []
        for i in range(self.nr_hyperparameters):
            hp_config = []
            config = self.get_benchmark_config(config_id=i)
            for hp_name in hp_names:
                hp_config.append(config[hp_name])
            hp_configs.append(hp_config)

        hp_configs = np.array(hp_configs)

        return hp_configs

    def get_best_performance(self):
        incumbent_curve = self.get_incumbent_curve()
        best_value = min(incumbent_curve)

        return best_value

    def get_benchmark_config(self, config_id):
        column_name = self.config_ids[config_id]
        config = {}
        for i, hp_name in enumerate(self.hp_names):
            config[hp_name] = column_name[i]
        return config

    def get_benchmark_curve(self, config_id):
        column_name = self.config_ids[config_id]
        data = self.benchmark_data.loc[:, column_name]
        data = data.iloc[:, 0].to_list()
        return data

    def get_curve_best(self, hp_index: int) -> float:
        curve = self.get_curve(hp_index, self.max_budget)
        best_value = min(curve)
        return best_value

    def get_performance(self, hp_index: int, budget: int) -> float:
        val_curve = self.get_benchmark_curve(config_id=hp_index)
        # val_curve = val_curve[1:]
        budget = int(budget)

        return val_curve[budget - 1]

    def get_curve(self, hp_index: int, budget: int) -> List[float]:
        val_curve = self.get_benchmark_curve(config_id=hp_index)
        # val_curve = val_curve[1:]
        budget = int(budget)

        return val_curve[0:budget]

    def get_incumbent_curve(self):
        val_curve = self.get_benchmark_curve(config_id=0)

        return val_curve

    def get_incumbent_config_id(self):
        return 0
