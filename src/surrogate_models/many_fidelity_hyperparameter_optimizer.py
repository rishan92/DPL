import sys
from copy import deepcopy
import os
import time
from typing import List, Tuple, Dict, Optional, Any, Union, Type
import pandas as pd
from numpy.typing import NDArray
from loguru import logger
import numpy as np
import random
from scipy.stats import norm
import torch
from types import SimpleNamespace
import wandb
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import copy

from src.benchmarks.base_benchmark import BaseBenchmark
from src.dataset.tabular_dataset import TabularDataset
from src.models.power_law.ensemble_model import EnsembleModel
from src.data_loader.surrogate_data_loader import SurrogateDataLoader
from src.models.deep_kernel_learning.dyhpo_model import DyHPOModel
import global_variables as gv
from src.history.history_manager import HistoryManager
from src.surrogate_models.base_hyperparameter_optimizer import BaseHyperparameterOptimizer
from src.plot.utils import plot_line
import src.models.activation_functions
from src.utils.utils import get_class_from_package, get_class_from_packages, get_inverse_function_class, \
    numpy_to_torch_apply
from scipy.stats import spearmanr
import properscoring as ps
from src.surrogate_models.hyperparameter_optimizer import HyperparameterOptimizer
from src.surrogate_models.asha import AHBOptimizer
from src.surrogate_models.dehb.interface import DEHBOptimizer
from src.surrogate_models.random_search import RandomOptimizer


class MFHyperparameterOptimizer(BaseHyperparameterOptimizer):
    surrogate_types = {
        'power_law': HyperparameterOptimizer,
        'dyhpo': HyperparameterOptimizer,
        'asha': AHBOptimizer,
        'dehb': DEHBOptimizer,
        # 'dragonfly': DragonFlyOptimizer,
        'random': RandomOptimizer,
    }

    def __init__(
        self,
        **kwargs
    ):
        super().__init__()

        self.surrogate_name = kwargs['surrogate_name']
        self.benchmark = kwargs['benchmark']
        self.fantasize_step = kwargs['fantasize_step']
        self.max_budgets = kwargs['max_budgets']
        self.min_budgets = kwargs['min_budgets']
        self.minimization = kwargs['minimization']
        self.fidelity_names = self.benchmark.fidelity_names
        self.previous_budgets = {}

        self.hp_optimizers: List[BaseHyperparameterOptimizer] = []
        self.hp_optimizer_class = self.surrogate_types[kwargs['surrogate_name']]
        for fidelity_name in self.fidelity_names:
            modified_kwarg = copy.copy(kwargs)
            modified_kwarg['max_budgets'] = self.max_budgets
            modified_kwarg['min_budgets'] = self.min_budgets
            modified_kwarg['fantasize_step'] = self.fantasize_step
            modified_kwarg['fidelity_name'] = fidelity_name
            hp_optimizer = self.hp_optimizer_class(**modified_kwarg)
            self.hp_optimizers.append(hp_optimizer)

    @staticmethod
    def get_default_meta(model_class):
        return {}

    @classmethod
    def set_meta(cls, config=None, **kwargs):
        pass

    def suggest(self) -> Tuple[List, List]:
        suggested_hp_indices = []
        suggested_budgets = []
        for i, hp_optimizer in enumerate(self.hp_optimizers):
            suggested_hp_index, budget = hp_optimizer.suggest()
            budgets = copy.copy(self.previous_budgets)
            budgets[self.fidelity_names[i]] = budget
            suggested_hp_indices.append(suggested_hp_index)
            suggested_budgets.append(budgets)
        if len(self.previous_budgets) == 0:
            self.previous_budgets = {k: v for b in suggested_budgets for k, v in b.items()}
            suggested_budgets = [self.previous_budgets] * len(suggested_budgets)
        return suggested_hp_indices, suggested_budgets

    def observe(self, hp_index: List[int], budget: List[Dict], hp_curve: List[float]):
        if not self.minimization:
            best_index = np.argmax(hp_curve)
        else:
            best_index = np.argmin(hp_curve)

        best_hp_index = hp_index[best_index]
        best_budget = budget[best_index]
        best_hp_curve = hp_curve[best_index]
        self.previous_budgets = best_budget

        for i, hp_optimizer in enumerate(self.hp_optimizers):
            hp_optimizer.observe(
                hp_index=best_hp_index, budget=best_budget, hp_curve=best_hp_curve
            )

    def plot_pred_curve(self, hp_index, benchmark, surrogate_budget, output_dir, prefix=""):
        for i, hp_optimizer in enumerate(self.hp_optimizers):
            hp_optimizer.plot_pred_curve(
                hp_index=hp_index,
                benchmark=benchmark,
                surrogate_budget=surrogate_budget,
                output_dir=output_dir,
                prefix=prefix
            )
