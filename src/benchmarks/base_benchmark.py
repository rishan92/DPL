from abc import ABC, abstractmethod
from typing import List, Union, Dict, Tuple
from pathlib import Path
import numpy as np
from collections import OrderedDict
# from numpy.typing import NDArray
import ConfigSpace as CS
from ConfigSpace.hyperparameters import FloatHyperparameter, IntegerHyperparameter, Constant, \
    CategoricalHyperparameter, OrdinalHyperparameter


class BaseBenchmark(ABC):
    nr_hyperparameters = None
    max_budget = None
    min_budget = None
    log_indicator = None
    hp_names = None
    param_space = None
    # if the best value corresponds to a lower value
    minimization_metric = True
    max_budgets = None

    def __init__(self, path_to_json_file: Path, seed=0):
        self.path_to_json_file: Path = path_to_json_file
        self.max_value = None
        self.min_value = None
        self.categorical_indicator = None
        self.objective_performance_info = {}
        self.categories = None
        self.seed = seed
        self.incumbent_curve = None
        self.incumbent_config_id = None
        self.worst_curve = None
        self.worst_config_id = None
        self.nr_hyperparameters = None
        self.max_budgets = None

    def load_dataset_names(self):
        raise NotImplementedError('Please implement the load_dataset_names method')

    @abstractmethod
    def get_hyperparameter_candidates(self) -> np.ndarray:
        raise NotImplementedError('Please extend the get_hyperparameter_candidates method')

    @abstractmethod
    def get_performance(self, hp_index: int, fidelity_id: Tuple[int]) -> float:
        raise NotImplementedError('Please extend the get_performance method')

    @abstractmethod
    def get_curve(self, hp_index: int, budget: Union[int, Dict]) -> List[float]:
        raise NotImplementedError

    @property
    def is_minimize(self) -> bool:
        return self.minimization_metric

    def size(self) -> int:
        return self.nr_hyperparameters * self.max_budget

    def get_objective_function_performance(self, hp_index: int, fidelity_id: Tuple[int]) -> Tuple[List, List]:
        performance = self.get_performance(hp_index=hp_index, fidelity_id=fidelity_id)
        return [performance], [fidelity_id]

    def close(self):
        pass

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

    def get_best_performance(self):
        incumbent_curve = self.get_incumbent_curve()
        if self.is_minimize:
            best_value = min(incumbent_curve)
        else:
            best_value = max(incumbent_curve)

        return best_value

    def get_curve_best(self, hp_index: int) -> float:
        curve, _ = self.get_curve(hp_index, self.max_budget)
        if self.is_minimize:
            best_value = min(curve)
        else:
            best_value = max(curve)
        return best_value

    def calc_benchmark_stats(self):
        max_value = np.NINF
        max_curve = None
        max_id = None
        min_value = np.PINF
        min_curve = None
        min_id = None
        for index in range(self.nr_hyperparameters):
            val_curve, _ = self.get_curve(hp_index=index, budget=self.max_budgets)
            value = min(val_curve)
            if value < min_value:
                min_value = value
                min_curve = val_curve
                min_id = index
            value = max(val_curve)
            if value > max_value:
                max_value = value
                max_curve = val_curve
                max_id = index

        if self.is_minimize:
            self.incumbent_curve = min_curve
            self.worst_curve = max_curve
            self.incumbent_config_id = min_id
            self.worst_config_id = max_id
        else:
            self.incumbent_curve = max_curve
            self.worst_curve = min_curve
            self.incumbent_config_id = max_id
            self.worst_config_id = min_id

    def get_incumbent_curve(self):
        if self.incumbent_curve is None:
            self.calc_benchmark_stats()
        return self.incumbent_curve

    def get_worst_performance(self):
        if self.worst_curve is None:
            self.calc_benchmark_stats()
        return self.worst_curve

    def get_incumbent_config_id(self):
        if self.incumbent_config_id is None:
            self.calc_benchmark_stats()
        return self.incumbent_config_id
