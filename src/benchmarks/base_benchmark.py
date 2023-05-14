from abc import ABC, abstractmethod
from typing import List, Union
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class BaseBenchmark(ABC):
    nr_hyperparameters = None
    max_budget = None
    min_budget = None
    log_indicator = None
    hp_names = None
    param_space = None
    # if the best value corresponds to a lower value
    minimization_metric = True

    def __init__(self, path_to_json_file: Path):
        self.path_to_json_file: Path = path_to_json_file
        self.max_value = None
        self.min_value = None
        self.categorical_indicator = None

    def _load_benchmark(self):
        raise NotImplementedError('Please implement the load_benchmark method')

    def load_dataset_names(self):
        raise NotImplementedError('Please implement the load_dataset_names method')

    @abstractmethod
    def get_hyperparameter_candidates(self) -> NDArray:
        raise NotImplementedError('Please extend the get_hyperparameter_candidates method')

    @abstractmethod
    def get_performance(self, hp_index: int, budget: int) -> float:
        raise NotImplementedError('Please extend the get_performance method')

    @abstractmethod
    def get_incumbent_config_id(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_best_performance(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_curve(self, hp_index: int, budget: int) -> List[float]:
        raise NotImplementedError

    def get_curve_best(self, hp_index: int) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_incumbent_curve(self) -> List[float]:
        raise NotImplementedError

    @property
    def is_minimize(self) -> bool:
        return self.minimization_metric

    def size(self) -> int:
        return self.nr_hyperparameters * self.max_budget
