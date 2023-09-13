from typing import List, Tuple, Union, Dict
from abc import ABC, abstractmethod

from src.models.base.meta import Meta


class BaseHyperparameterOptimizer(Meta, ABC):
    @abstractmethod
    def suggest(self) -> Tuple[int, int]:
        raise NotImplementedError

    @abstractmethod
    def observe(self, hp_index: int, budget: Union[int, Dict], hp_curve: float):
        raise NotImplementedError

    @classmethod
    def set_meta(cls, config=None, **kwargs):
        pass

    @staticmethod
    def get_default_meta(**kwargs):
        return {}

    def plot_pred_curve(self, hp_index, benchmark, surrogate_budget, output_dir, prefix=""):
        pass

    def plot_pred_dist(self, benchmark, surrogate_budget, output_dir, prefix=""):
        pass

    @property
    def meta_predict_fidelity_mode(self):
        return "next"
