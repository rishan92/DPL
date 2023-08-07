from copy import deepcopy
import os
import time
from typing import List, Tuple, Dict, Optional, Any, Union
from loguru import logger
import numpy as np
import random
from scipy.stats import norm
import torch
from types import SimpleNamespace
import global_variables as gv
import pandas as pd
import copy

from src.dataset.tabular_dataset import TabularDataset
import functools
from functools import partial
# from numpy.typing import NDArray
from src.utils.utils import get_class_from_package, get_class_from_packages, numpy_to_torch_apply
import src.models.activation_functions
from src.benchmarks.base_benchmark import BaseBenchmark
import matplotlib.pyplot as plt
from src.history.fidelity_manager import FidelityManager


class HistoryManager:
    def __init__(self, hp_candidates, max_budgets, min_budgets, fantasize_step, use_learning_curve,
                 use_learning_curve_mask, fidelity_manager: FidelityManager = None, fill_value='zero',
                 use_target_normalization=False, use_scaled_budgets=True,
                 model_output_normalization=None, cnn_kernel_size=0, target_normalization_range=None,
                 use_sample_weights=False, use_sample_weight_by_budget=False, sample_weight_by_budget_strategy=None,
                 use_sample_weight_by_label=False, use_y_constraint_weights=False, fidelity_names=None):
        assert fill_value in ["zero", "last"], "Invalid fill value mode"
        # assert predict_mode in ["end_budget", "next_budget"], "Invalid predict mode"
        # assert curve_size_mode in ["fixed", "variable"], "Invalid curve size mode"
        self.hp_candidates = hp_candidates
        self.max_budgets = max_budgets
        self.min_budgets = min_budgets
        self.fill_value = fill_value
        self.use_learning_curve = use_learning_curve
        self.use_learning_curve_mask = use_learning_curve_mask
        self.use_scaled_budgets = use_scaled_budgets
        self.cnn_kernel_size = cnn_kernel_size
        self.use_sample_weights = use_sample_weights
        self.use_sample_weight_by_budget = use_sample_weight_by_budget
        self.sample_weight_by_budget_strategy = sample_weight_by_budget_strategy
        self.use_sample_weight_by_label = use_sample_weight_by_label
        self.use_y_constraint_weights = use_y_constraint_weights
        self.fidelity_names = fidelity_names
        self.extra_budgets_names = list(self.max_budgets.keys())
        self.extra_budgets_names = [v for v in self.extra_budgets_names if v not in self.fidelity_names]
        self.fidelity_manager = fidelity_manager

        self.use_target_normalization = use_target_normalization
        self.target_normalization_range = \
            target_normalization_range if target_normalization_range is not None else [0, 1]
        self.target_normalization_fn = None
        self.max_curve_value = 0
        self.target_normalization_value = 1

        self.model_output_normalization = model_output_normalization
        self.model_output_normalization_fn = None
        if self.model_output_normalization and self.model_output_normalization != "Identity":
            torch_function = get_class_from_packages([torch.nn, src.models.activation_functions],
                                                     self.model_output_normalization)()
            self.model_output_normalization_fn = partial(numpy_to_torch_apply, torch_function=torch_function)

        # the keys will be hyperparameter indices while the value
        # will be a list with all the budgets evaluated for examples
        # and with all performances for the performances
        self.examples: Dict[int, List[int]] = dict()
        self.performance_history: Dict[int, List[float]] = dict()
        self.fidelity_id_history: Dict[int, List[Tuple[int]]] = dict()
        self.extra_budget: Dict[int, List[Dict]] = dict()

        self.last_point = None

        self.fantasize_step = fantasize_step

        self.is_train_data_modified = True
        self.cached_train_dataset = None
        self.is_test_data_modified = True
        self.cached_test_dataset = None

    def set_target_normalization_value(self):
        self.target_normalization_value = self.max_curve_value
        gap = self.target_normalization_range[1] - self.target_normalization_range[0]
        self.target_normalization_fn = lambda x: (x * gap / self.target_normalization_value) + \
                                                 self.target_normalization_range[0]
        return self.target_normalization_value

    def get_initial_empty_value(self):
        initial_empty_value = self.get_mean_initial_value() if self.fill_value == 'last' else 0
        return initial_empty_value

    def add(self, hp_index: int, fidelity_id: List[Tuple[int]], hp_curve: List):
        extra_b = None
        # if isinstance(b, Dict):
        #     fidelity_size = len(self.fidelity_names)
        #     in_fidelity_size = 1 if not isinstance(b, Dict) else len(b)
        #     if fidelity_size < in_fidelity_size:
        #         extra_b = {k: v for k, v in b.items() if k not in self.fidelity_names}
        #         b = {k: b[k] for k in self.fidelity_names}
        #
        #         if hp_index in self.extra_budget:
        #             self.extra_budget[hp_index].extend(extra_b)
        #         else:
        #             self.extra_budget[hp_index] = [extra_b]

        b = self.fidelity_manager.get_fidelities(fidelity_ids=fidelity_id)
        if hp_index in self.performance_history:
            self.performance_history[hp_index].extend(hp_curve)
            self.examples[hp_index].extend(b)
            self.fidelity_id_history[hp_index].extend(fidelity_id)
        else:
            self.performance_history[hp_index] = hp_curve
            self.examples[hp_index] = b
            self.fidelity_id_history[hp_index] = fidelity_id

        self.fidelity_manager.set_fidelity_id(configuration_id=hp_index, fidelity_id=fidelity_id[-1])

        initial_empty_value = self.get_initial_empty_value()

        cur_curve = self.performance_history[hp_index]

        self.last_point = (
            hp_index,
            fidelity_id[-1],
            cur_curve[-1],
            cur_curve[0:-1] if len(cur_curve) > 1 else [initial_empty_value],
            extra_b
        )

        max_curve = np.max(cur_curve)
        self.max_curve_value = max(self.max_curve_value, max_curve)
        # self.max_curve_value = 10

        self.is_train_data_modified = True
        self.is_test_data_modified = True

    def get_evaluated_budgets(self, suggested_hp_index):
        if suggested_hp_index in self.examples:
            eval_budgets = self.examples[suggested_hp_index]
            return eval_budgets
        else:
            return []

    def get_evaluted_indices(self):
        if len(self.examples) == 0:
            return []
        else:
            return list(self.examples.keys())

    def get_last_sample(self, curve_size_mode):
        newp_index, newp_budget, newp_performance, newp_curve, newp_extra_b = deepcopy(self.last_point)

        modified_curve = self.get_processed_curves(curves=[newp_curve], curve_size_mode=curve_size_mode,
                                                   real_budgets=newp_budget)

        new_example = torch.tensor(self.hp_candidates[newp_index], dtype=torch.float32)
        new_example = torch.unsqueeze(new_example, dim=0)
        if newp_extra_b is not None:
            for k in self.extra_budgets_names:
                newp_extra_b[k] = newp_extra_b[k] / self.max_budgets[k]
            train_extra_budgets = np.array([newp_extra_b[k] for k in self.extra_budgets_names])
            train_extra_budgets = np.expand_dims(train_extra_budgets, axis=0)
            new_example = np.concatenate([new_example, train_extra_budgets], axis=1)

        # if self.use_scaled_budgets:
        #     # newp_budget = newp_budget / self.max_budgets[self.fidelity_name]
        #     for k in self.fidelity_names:
        #         newp_budget[k] = newp_budget[k] / self.max_budgets[k]

        # newp_budget = torch.tensor([newp_budget[k] for k in self.fidelity_names], dtype=torch.float32)
        newp_budget = self.fidelity_manager.get_fidelities(
            fidelity_ids=newp_budget, is_normalized=self.use_scaled_budgets
        )
        newp_budget = np.array(newp_budget)
        newp_budget = torch.from_numpy(newp_budget)
        newp_budget = torch.unsqueeze(newp_budget, dim=0)

        if self.use_target_normalization:
            newp_performance = self.target_normalization_fn(newp_performance)

        if self.model_output_normalization_fn:
            newp_performance = self.model_output_normalization_fn(np.array(newp_performance)).item()

        newp_performance = torch.tensor([newp_performance], dtype=torch.float32)

        if modified_curve is not None:
            modified_curve = torch.from_numpy(modified_curve)
        else:
            modified_curve = torch.tensor([0], dtype=torch.float32)
            modified_curve = torch.unsqueeze(modified_curve, dim=0)

        if self.cached_train_dataset is not None and \
            (self.use_sample_weights or self.use_sample_weight_by_budget or
             self.use_sample_weight_by_label or self.use_y_constraint_weights):
            weight = self.cached_train_dataset.get_weight(x=new_example, budget=newp_budget)
        else:
            weight = torch.tensor([1.0])

        last_sample = (new_example, newp_performance, newp_budget, modified_curve, weight)
        return last_sample

    def history_configurations(self, curve_size_mode) -> \
        Tuple[np.ndarray, np.ndarray, List[Tuple[int]], Optional[np.ndarray],
              np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Generate the configurations, labels, budgets and curves
        based on the history of evaluated configurations.

        Returns:
            (train_examples, train_labels, train_budgets, train_curves): Tuple
                A tuple of examples, labels and budgets for the
                configurations evaluated so far.
        """
        train_indices = []
        train_labels = []
        train_budgets = []
        train_curves = []
        train_weights = []
        train_max_budget = []
        train_extra_budget = []
        initial_empty_value = self.get_initial_empty_value()

        if self.sample_weight_by_budget_strategy is not None:
            if self.sample_weight_by_budget_strategy.isdigit():
                power_value = int(self.sample_weight_by_budget_strategy)
                weight_fn = lambda w: np.power(w / self.max_budgets[self.fidelity_name], power_value)
            elif self.sample_weight_by_budget_strategy == "softmax":
                weight_fn = lambda w: np.exp(w - np.max(w))
            else:
                raise NotImplementedError
        else:
            weight_fn = lambda w: w

        for hp_index in self.examples:
            budgets = self.examples[hp_index]
            budgets = self.fidelity_id_history[hp_index]
            performances = self.performance_history[hp_index]

            # weights = np.array(budgets, dtype=np.float32)
            # weights = weight_fn(weights)
            # weights /= weights.sum()
            # weights *= weights.shape[0]

            for i, (budget, performance) in enumerate(zip(budgets, performances)):
                train_indices.append(hp_index)
                train_budgets.append(budget)
                train_labels.append(performance)
                if self.use_learning_curve:
                    train_curve = performances[:i] if i > 0 else [initial_empty_value]
                    train_curves.append(train_curve)
                # train_weights.append(weights[i])
                # train_max_budget.append(budgets[-1])
                if len(self.extra_budgets_names) != 0:
                    train_extra_budget.append(self.extra_budget[hp_index][i])

        train_curves = self.get_processed_curves(curves=train_curves, curve_size_mode=curve_size_mode,
                                                 real_budgets=train_budgets)

        train_indices = np.array(train_indices, dtype=int)
        train_labels = np.array(train_labels, dtype=np.float32)
        train_weights = np.array(train_weights, dtype=np.float32)
        # train_max_budget = np.array(train_max_budget, dtype=np.float32)
        train_max_budget = None

        # train_budgets_indices = pd.DataFrame(train_budgets, columns=self.fidelity_names).astype(int)
        train_budgets_indices = train_budgets

        if len(self.extra_budgets_names) != 0:
            train_extra_budgets = pd.DataFrame(train_extra_budget, columns=self.extra_budgets_names).astype(np.float32)
        else:
            train_extra_budgets = None

        return train_indices, train_labels, train_budgets_indices, \
               train_curves, train_weights, train_max_budget, train_extra_budgets

    def get_train_dataset(self, curve_size_mode) -> TabularDataset:
        """This method is called to prepare the necessary training dataset
        for training a model.

        Returns:
            train_dataset: A dataset consisting of examples, labels, budgets
                and learning curves.
        """
        if not self.is_train_data_modified:
            return self.cached_train_dataset

        hp_indices, train_labels, train_fidelity_ids, train_curves, \
        train_weights, train_max_budget, train_extra_budgets = self.history_configurations(curve_size_mode)

        train_budgets = self.fidelity_manager.get_fidelities(
            fidelity_ids=train_fidelity_ids, is_normalized=self.use_scaled_budgets
        )
        # train_budgets_pd = pd.DataFrame(train_budgets, columns=self.fidelity_names).astype(np.float32)
        # if self.use_scaled_budgets:
        #     # scale budgets to [0, 1]
        #     # train_budgets = train_budgets / self.max_budgets[self.fidelity_name]
        #     for col in train_budgets_pd.columns:
        #         train_budgets_pd[col] = train_budgets_pd[col] / self.max_budgets[col]
        # train_budgets = train_budgets_pd.to_numpy()
        train_budgets = np.array(train_budgets, dtype=np.float32)

        if self.use_target_normalization:
            train_labels = self.target_normalization_fn(train_labels)

        if self.model_output_normalization_fn:
            train_labels = self.model_output_normalization_fn(train_labels)

        all_weights = np.ones_like(train_weights)

        if self.use_sample_weight_by_budget:
            all_weights = train_weights

        if self.use_sample_weight_by_label:
            power = 1
            if isinstance(self.use_sample_weight_by_label, int):
                power = self.use_sample_weight_by_label
            weights = train_labels.copy()
            # max_weight = np.max(weights)
            # min_weight = np.min(weights)
            # if max_weight != min_weight:
            #     weights = (weights - min_weight) / (max_weight - min_weight)
            # weights = np.abs(np.exp(-power * weights) - np.exp(-power)) / (1 - np.exp(-power))
            weights = 1 / (weights + 1e-3)
            weights *= weights.shape[0] / weights.sum()

            all_weights = weights * all_weights
            all_weights *= all_weights.shape[0] / all_weights.sum()

        train_weights = all_weights

        if self.use_y_constraint_weights:
            if isinstance(self.use_y_constraint_weights, int):
                power = self.use_y_constraint_weights
                y_constraint_weights = train_max_budget / self.max_budgets[self.fidelity_name]
                y_constraint_weights = np.abs(np.exp(-power * y_constraint_weights) - np.exp(-power)) / (
                    1 - np.exp(-power))
            else:
                y_constraint_weights = 1 - train_max_budget / self.max_budgets[self.fidelity_name]

            if self.use_sample_weight_by_budget or self.use_sample_weight_by_label:
                train_weights = np.expand_dims(train_weights, axis=1)
                y_constraint_weights = np.expand_dims(y_constraint_weights, axis=1)
                train_weights = np.concatenate([train_weights, y_constraint_weights], axis=1)
            else:
                train_weights = y_constraint_weights

        # This creates a copy
        train_examples = self.hp_candidates[hp_indices]
        if train_extra_budgets is not None:
            for col in train_extra_budgets.columns:
                train_extra_budgets[col] = train_extra_budgets[col] / self.max_budgets[col]
            train_extra_budgets = train_extra_budgets.to_numpy()
            train_extra_budgets = train_extra_budgets.astype(np.float32)
            train_examples = np.concatenate([train_examples, train_extra_budgets], axis=1)

        train_examples = torch.from_numpy(train_examples)
        train_labels = torch.from_numpy(train_labels)
        train_budgets = torch.from_numpy(train_budgets)
        train_curves = torch.from_numpy(train_curves) if train_curves is not None else None

        if self.use_sample_weight_by_budget or self.use_sample_weight_by_label or self.use_y_constraint_weights:
            train_weights = torch.from_numpy(train_weights)
        else:
            train_weights = None

        train_dataset = TabularDataset(
            X=train_examples,
            Y=train_labels,
            budgets=train_budgets,
            curves=train_curves,
            use_sample_weights=self.use_sample_weights,
            use_sample_weight_by_budget=self.use_sample_weight_by_budget,
            weights=train_weights
        )

        self.cached_train_dataset = train_dataset
        self.is_train_data_modified = False

        return train_dataset

    def get_predict_curves_dataset(self, hp_index, curve_size_mode, fidelity_name='epochs'):
        curves = []
        real_budgets = []
        initial_empty_value = self.get_initial_empty_value()
        first_budgets = {fidelity_name: min(self.fantasize_step[fidelity_name], self.min_budgets[fidelity_name])
                         for fidelity_name in self.fidelity_names}

        if hp_index in self.examples:
            budgets: List = self.examples[hp_index]
            budgets = self.fidelity_id_history[hp_index]
            max_train_fidelity_id = budgets[-1]
            performances = self.performance_history[hp_index]
            for i, (budget, performance) in enumerate(zip(budgets, performances)):
                real_budgets.append(budget)
                train_curve = performances[:i] if i > 0 else [initial_empty_value]
                curves.append(train_curve)
        else:
            max_train_fidelity_id = self.fidelity_manager.first_fidelity_id
            real_budgets.append(first_budgets)
            curves.append([0])

        curves = self.get_processed_curves(curves=curves, curve_size_mode=curve_size_mode, real_budgets=real_budgets)

        # real_budgets = np.arange(
        #     self.min_budgets[fidelity_name],
        #     self.max_budgets[fidelity_name] + self.fantasize_step[fidelity_name],
        #     self.fantasize_step[fidelity_name]
        # )
        # real_budgets = np.around(real_budgets, decimals=4)
        # real_budgets = [{**max_train_budget, fidelity_name: v} for v in real_budgets]

        # budgets_pd = pd.DataFrame(real_budgets, columns=self.fidelity_names).astype(np.float32)
        #
        # if self.use_scaled_budgets:
        #     for col in budgets_pd.columns:
        #         budgets_pd[col] = budgets_pd[col] / self.max_budgets[col]
        # p_budgets = budgets_pd.to_numpy()

        fidelity_index = self.fidelity_manager.fidelity_names.index(fidelity_name)
        fidelity = self.fidelity_manager.fidelity_space[fidelity_name]
        real_budgets_id = []
        max_train_fidelity_ids = list(max_train_fidelity_id)
        for i in range(len(fidelity)):
            max_train_fidelity_ids[fidelity_index] = i
            real_budgets_id.append(tuple(max_train_fidelity_ids))

        max_train_fidelity = self.fidelity_manager.get_fidelities(
            fidelity_ids=max_train_fidelity_id, is_normalized=False, return_dict=True,
        )

        p_budgets = self.fidelity_manager.get_fidelities(
            fidelity_ids=real_budgets_id, is_normalized=self.use_scaled_budgets
        )
        p_budgets = np.array(p_budgets, dtype=np.float32)
        p_budgets = torch.from_numpy(p_budgets)

        # real_budgets = self.fidelity_manager.get_fidelities(
        #     fidelity_ids=real_budgets_id, is_normalized=False, return_dict=True,
        # )

        fidelity_steps = len(real_budgets_id)
        p_config = self.hp_candidates[hp_index]
        p_config = torch.tensor(p_config, dtype=torch.float32)
        p_config = p_config.expand(fidelity_steps, -1)

        p_curve = None
        if curves is not None:
            p_curve = torch.tensor(curves, dtype=torch.float32)
            p_curve_last_row = p_curve[-1].unsqueeze(0)
            p_curve_num_repeats = fidelity_steps - p_curve.size(0)
            repeated_last_row = p_curve_last_row.repeat_interleave(p_curve_num_repeats, dim=0)
            p_curve = torch.cat((p_curve, repeated_last_row), dim=0)

        pred_test_data = TabularDataset(
            X=p_config,
            budgets=p_budgets,
            curves=p_curve
        )

        return pred_test_data, real_budgets_id, max_train_fidelity

    def get_processed_curves(self, curves, curve_size_mode, real_budgets) -> Optional[np.ndarray]:
        if self.use_learning_curve:
            if curve_size_mode == "variable":
                min_size = self.cnn_kernel_size
            elif curve_size_mode == "fixed":
                min_size = self.max_budgets[self.fidelity_name] - 1
            else:
                raise NotImplementedError

            curves = self.patch_curves_to_same_length(curves=curves, min_size=min_size)

            if self.use_learning_curve_mask:
                curves = self.add_curve_missing_value_mask(curves, real_budgets)
        else:
            curves = None
        return curves

    def patch_curves_to_same_length(self, curves: List[List[float]], min_size: int) -> np.ndarray:
        """
        Patch the given curves to the same length.

        Finds the maximum curve length and patches all
        other curves that are shorter in length with zeroes.

        Args:
            curves: The given hyperparameter curves.

        Returns:
            curves: The updated array where the learning
                curves are of the same length.
        """
        max_curve_length = min_size
        for curve in curves:
            if len(curve) > max_curve_length:
                max_curve_length = len(curve)

        extended_curves = np.empty(shape=(len(curves), max_curve_length), dtype=np.float32)
        for i, curve in enumerate(curves):
            extended_curves[i, :len(curve)] = curve
            fill_value = curve[-1] if self.fill_value == 'last' else 0
            extended_curves[i, len(curve):] = fill_value

        return extended_curves

    def calculate_fidelity_ymax(self, fidelity: int) -> float:
        """Calculate the incumbent for a certain fidelity level.

        Args:
            fidelity: int
                The given budget fidelity.

        Returns:
            best_value: float
                The incumbent value for a certain fidelity level.
        """
        config_values = []
        for example_index in self.examples.keys():
            try:
                performance = self.performance_history[example_index][fidelity - 1]
            except IndexError:
                performance = self.performance_history[example_index][-1]
            config_values.append(performance)

        # lowest error corresponds to best value
        best_value = min(config_values)

        return best_value

    def calculate_fidelity_ymax_dyhpo(self, fidelity: Union[int, Dict]):
        """
        Find ymax for a given fidelity level.

        If there are hyperparameters evaluated for that fidelity
        take the maximum from their values. Otherwise, take
        the maximum from all previous fidelity levels for the
        hyperparameters that we have evaluated.

        Args:
            fidelity: The fidelity of the hyperparameter
                configuration.

        Returns:
            best_value: The best value seen so far for the
                given fidelity.
        """
        exact_fidelity_config_values = []
        lower_fidelity_config_values = []

        for example_index in self.examples.keys():
            fidelity_index = -1
            for i, entry in enumerate(self.examples[example_index]):
                if entry == fidelity:
                    fidelity_index = i
                    break
            if fidelity_index >= 0:
                performance = self.performance_history[example_index][fidelity_index]
                exact_fidelity_config_values.append(performance)
            else:
                learning_curve = self.performance_history[example_index]
                # The hyperparameter was not evaluated until fidelity, or more.
                # Take the maximum value from the curve.
                lower_fidelity_config_values.append(min(learning_curve))

        if len(exact_fidelity_config_values) > 0:
            # lowest error corresponds to best value
            best_value = min(exact_fidelity_config_values)
        else:
            best_value = min(lower_fidelity_config_values)

        return best_value

    def add_curve_missing_value_mask(self, curves: np.ndarray, budgets: List[int]) -> np.ndarray:
        missing_value_mask = self.prepare_missing_values_masks(budgets, curves.size(1))

        # add depth dimension to the train_curves array and missing_value_matrix
        curves = np.expand_dims(curves, 1)
        missing_value_mask = np.expand_dims(missing_value_mask, 1)
        curves = np.concatenate((curves, missing_value_mask), axis=1)

        return curves

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def get_curve_mask(budget: int, size: int):
        mask = np.zeros(shape=(size,), dtype=bool)
        mask[:budget] = True
        return mask

    def prepare_missing_values_masks(self, budgets: List[int], size: int) -> np.ndarray:
        missing_value_masks = []

        for i in range(len(budgets)):
            budget = budgets[i]
            budget = budget - 1
            budget = int(budget)

            mask = self.get_curve_mask(budget, size)
            missing_value_masks.append(mask)

        missing_value_masks = np.array(missing_value_masks, dtype=bool)
        return missing_value_masks

    def get_mean_initial_value(self):
        """Returns the mean initial value
        for all hyperparameter configurations in the history so far.

        Returns:
            mean_initial_value: float
                Mean initial value for all hyperparameter configurations
                observed.
        """
        first_values = []
        for performance_curve in self.performance_history.values():
            first_values.append(performance_curve[0])

        mean_initial_value = np.mean(first_values)

        return mean_initial_value

    def get_candidate_configurations_dataset(self, predict_mode, curve_size_mode) -> \
        Tuple[TabularDataset, np.ndarray, List[Dict]]:

        if not self.is_test_data_modified:
            return self.cached_test_dataset

        hp_indices, hp_fidelity_ids, real_fidelity_ids, hp_curves, hp_extra_budgets = \
            self.generate_candidate_configurations(
                predict_mode,
                curve_size_mode
            )

        train_fidelities = self.fidelity_manager.get_fidelities(
            fidelity_ids=hp_fidelity_ids, is_normalized=self.use_scaled_budgets
        )
        train_fidelities_pd = pd.DataFrame(train_fidelities, columns=self.fidelity_manager.fidelity_names).astype(
            np.float32)

        real_budgets = self.fidelity_manager.get_fidelities(
            fidelity_ids=real_fidelity_ids, is_normalized=False
        )

        # if self.use_scaled_budgets:
        #     # scale budgets to [0, 1]
        #     # budgets = budgets / self.max_budgets[self.fidelity_name]
        #     for col in budgets_pd.columns:
        #         budgets_pd[col] = budgets_pd[col] / self.max_budgets[col]
        # budgets = budgets_pd.to_numpy()
        budgets = np.array(train_fidelities, dtype=np.float32)

        # This creates a copy with required indices
        configurations = self.hp_candidates[hp_indices]
        if hp_extra_budgets is not None:
            for col in hp_extra_budgets.columns:
                hp_extra_budgets[col] = hp_extra_budgets[col] / self.max_budgets[col]
            hp_extra_budgets = hp_extra_budgets.to_numpy()
            hp_extra_budgets = hp_extra_budgets.astype(np.float32)
            configurations = np.concatenate([configurations, hp_extra_budgets], axis=1)

        configurations = torch.from_numpy(configurations)
        budgets = torch.from_numpy(budgets)
        hp_curves = torch.from_numpy(hp_curves) if hp_curves is not None else None

        test_data = TabularDataset(
            X=configurations,
            budgets=budgets,
            curves=hp_curves,
        )

        self.cached_test_dataset = (test_data, hp_indices, real_budgets)
        self.is_test_data_modified = False

        return test_data, hp_indices, real_budgets

    # TODO: break this function to only handle candidates in history and make config manager handle configs
    #  not in history
    def generate_candidate_configurations(self, predict_mode, curve_size_mode) -> \
        Tuple[np.ndarray, List[Tuple[int]], List[Tuple[int]], Optional[np.ndarray], pd.DataFrame]:
        """Generate candidate configurations that will be
        fantasized upon.

        Returns:
            (configurations, hp_indices, hp_budgets, real_budgets, hp_curves): Tuple
                A tuple of configurations, their indices in the hp list,
                the budgets that they should be fantasized upon, the maximal
                budgets they have been evaluated and their corresponding performance
                curves.
        """
        hp_indices = []
        hp_budgets = []
        hp_curves = []
        real_budgets = []
        real_fidelity_ids = []
        real_extra_budgets = []
        hp_extra_budgets = []
        initial_empty_value = self.get_mean_initial_value() if self.fill_value == 'last' else 0

        first_budgets = {fidelity_name: self.min_budgets[fidelity_name] for fidelity_name in self.fidelity_names}
        first_extra_budgets = {fidelity_name: min(self.fantasize_step[fidelity_name], self.min_budgets[fidelity_name])
                               for fidelity_name in self.extra_budgets_names}
        max_budgets = {fidelity_name: self.max_budgets[fidelity_name] for fidelity_name in self.fidelity_names}
        max_extra_budgets = {fidelity_name: self.max_budgets[fidelity_name] for fidelity_name in
                             self.extra_budgets_names}

        for hp_index in range(0, self.hp_candidates.shape[0]):
            if hp_index in self.examples:
                budgets: List = self.examples[hp_index]
                # Take the max budget evaluated for a certain hpc
                max_budget: Dict = budgets[-1]
                max_budget = dict(zip(self.fidelity_manager.fidelity_names, max_budget))
                num_max_budgets = 0
                next_budget = {}
                for k in self.fidelity_names:
                    next_b = max_budget[k] + self.fantasize_step[k]
                    next_b = round(next_b, 4)
                    if next_b >= self.max_budgets[k]:
                        next_b = self.max_budgets[k]
                        num_max_budgets += 1
                    next_budget[k] = next_b
                if num_max_budgets == len(self.fidelity_names):
                    continue
                # budgets: List = self.fidelity_id_history[hp_index]
                # next_fidelity_id = self.fidelity_manager.get_next_fidelity_id(configuration_id=hp_index)
                # if next_fidelity_id is None:
                #     continue
                real_budgets.append(next_budget)
                if len(self.extra_budgets_names) != 0:
                    real_extra_budgets.append(self.extra_budget[hp_index][-1])
                learning_curve = self.performance_history[hp_index]

                budget_index = len(budgets)  # - 1
                hp_curve = learning_curve[:budget_index] if budget_index > 0 else [initial_empty_value]
            else:
                real_budgets.append(first_budgets)
                hp_curve = [initial_empty_value]
                if len(self.extra_budgets_names) != 0:
                    real_extra_budgets.append(first_extra_budgets)

            next_fidelity_id = self.fidelity_manager.get_next_fidelity_id(configuration_id=hp_index)
            if next_fidelity_id is None:
                continue
            real_fidelity_ids.append(next_fidelity_id)

            hp_indices.append(hp_index)
            hp_curves.append(hp_curve)
        hp_budgets = [max_budgets] * len(real_budgets)
        hp_fidelity_ids = [self.fidelity_manager.last_fidelity_id] * len(real_budgets)
        hp_extra_budgets = [max_extra_budgets] * len(real_budgets)

        hp_curves = self.get_processed_curves(curves=hp_curves, curve_size_mode=curve_size_mode,
                                              real_budgets=hp_budgets)

        if predict_mode == "next_budget":
            # make sure there is a copy happening because hp_budgets get normalized and real_budgets does not.
            # Creating np.array below copies the data.
            hp_budgets = real_budgets
            hp_fidelity_ids = real_fidelity_ids
            hp_extra_budgets = real_extra_budgets

        hp_indices = np.array(hp_indices, dtype=int)
        hp_budgets = pd.DataFrame(hp_budgets, columns=self.fidelity_names).astype(np.float32)
        if len(self.extra_budgets_names) != 0:
            hp_extra_budgets = pd.DataFrame(hp_extra_budgets, columns=self.extra_budgets_names).astype(np.float32)
        else:
            hp_extra_budgets = None

        return hp_indices, hp_fidelity_ids, real_fidelity_ids, hp_curves, hp_extra_budgets

    def all_configurations(self, curve_size_mode, benchmark: BaseBenchmark) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:

        train_indices = []
        train_labels = []
        train_budgets = []
        train_curves = []
        is_up_curve = []
        best_labels = []
        train_max_budget = []
        initial_empty_value = self.get_initial_empty_value()

        budgets = range(1, self.max_budgets[self.fidelity_name] + 1)
        for hp_index in range(0, self.hp_candidates.shape[0]):
            performances = benchmark.get_curve(hp_index, budget=self.max_budgets[self.fidelity_name])
            if not benchmark.minimization_metric:
                performances = benchmark.max_value - np.array(performances)
                performances = performances.tolist()
            start_performance = performances[0]
            best_performance = min(performances)
            for budget in budgets:
                train_indices.append(hp_index)
                train_budgets.append(budget)
                train_labels.append(performances[budget - 1])
                is_up_curve.append(performances[budget - 1] > start_performance)
                best_labels.append(best_performance)
                train_max_budget.append(budgets[-1])
                if self.use_learning_curve:
                    train_curve = performances[:budget - 1] if budget > 0 else [initial_empty_value]
                    train_curves.append(train_curve)

        train_curves = self.get_processed_curves(curves=train_curves, curve_size_mode=curve_size_mode,
                                                 real_budgets=train_budgets)

        train_indices = np.array(train_indices, dtype=int)
        train_budgets = np.array(train_budgets, dtype=np.float32)
        train_labels = np.array(train_labels, dtype=np.float32)
        is_up_curve = np.array(is_up_curve, dtype=bool)
        best_labels = np.array(best_labels, dtype=np.float32)
        train_max_budget = np.array(train_max_budget, dtype=np.float32)

        return train_indices, train_labels, train_budgets, train_curves, is_up_curve, best_labels

    def get_check_train_validation_dataset(self, curve_size_mode, benchmark: BaseBenchmark,
                                           validation_configuration_ratio, validation_curve_ratio, validation_mode,
                                           check_model_train_mode, validation_curve_prob, seed) -> (
        TabularDataset, TabularDataset):
        """This method is called to prepare the necessary training dataset
        for training a model.

        Returns:
            train_dataset: A dataset consisting of examples, labels, budgets
                and learning curves.
        """

        indices = np.arange(self.hp_candidates.shape[0])
        np.random.seed(seed)
        train_indices = np.random.choice(indices,
                                         size=int(self.hp_candidates.shape[0] * (1 - validation_configuration_ratio)),
                                         replace=False)
        if len(train_indices) == self.hp_candidates.shape[0]:
            train_indices = []

        budgets = np.arange(1, self.max_budgets[self.fidelity_name] + 1)
        train_budgets_indices = budgets[:int(self.max_budgets[self.fidelity_name] * (1 - validation_curve_ratio)) + 1]
        val_budgets_indices = budgets[int(self.max_budgets[self.fidelity_name] * (1 - validation_curve_ratio)) + 1:]

        all_hp_indices, all_labels, all_budgets, all_curves, all_is_up_curve, all_best_labels = \
            self.all_configurations(curve_size_mode, benchmark=benchmark)

        self.max_curve_value = np.max(all_labels)
        target_normalization_value = self.set_target_normalization_value()

        train_hp_index = np.isin(all_hp_indices, train_indices)

        if check_model_train_mode == "exp":
            upper_budget = int(self.max_budgets[self.fidelity_name] * (1 - validation_curve_ratio)) + 1
            weights = 1.0 / budgets
            # weights /= np.sum(weights)
            weights[:upper_budget] = weights[:upper_budget] * validation_curve_prob / np.sum(weights[:upper_budget])
            weights[upper_budget:] = weights[upper_budget:] * (1 - validation_curve_prob) / np.sum(
                weights[upper_budget:])
            selected_budgets = np.random.choice(budgets, size=int(self.hp_candidates.shape[0]), p=weights)
            all_selected_budgets = selected_budgets[all_hp_indices]
            train_budget_index = all_budgets <= all_selected_budgets
        else:
            train_budget_index = np.isin(all_budgets, train_budgets_indices)

        train_hp_index = np.logical_and(train_hp_index, train_budget_index)

        val_hp_index = np.isin(all_budgets, val_budgets_indices)

        if validation_mode == "end" or validation_mode == "best":
            val_budget_index = all_budgets == self.max_budgets[self.fidelity_name]
            val_hp_index = np.logical_and(val_hp_index, val_budget_index)

        # a = val_hp_index.sum()
        # b = all_is_up_curve.sum()
        # filter out the curve points that goes up from the validation dataset
        all_is_down_curve = ~all_is_up_curve
        val_hp_index = np.logical_and(val_hp_index, all_is_down_curve)
        # c = val_hp_index.sum()

        # first_label_index = all_budgets == 1
        # first_label = all_labels[first_label_index]
        # last_label_index = all_budgets == self.max_budgets[self.fidelity_name]
        # last_label = all_best_labels[last_label_index]

        if self.use_scaled_budgets:
            # scale budgets to [0, 1]
            all_budgets = all_budgets / self.max_budgets[self.fidelity_name]

        transformed_all_labels = all_labels
        if self.use_target_normalization:
            transformed_all_labels = self.target_normalization_fn(transformed_all_labels)

        if self.model_output_normalization_fn:
            transformed_all_labels = self.model_output_normalization_fn(transformed_all_labels)

        # make train dataset
        train_hp_indices = all_hp_indices[train_hp_index]
        train_labels = transformed_all_labels[train_hp_index]
        train_budgets = all_budgets[train_hp_index]
        train_curves = all_curves[train_hp_index] if all_curves is not None else None

        max_dict = {i: np.max(train_budgets[train_hp_indices == i]) for i in np.unique(train_hp_indices)}
        train_max_budget = [max_dict[i] for i in train_hp_indices]
        train_max_budget = np.array(train_max_budget)

        # train_mu = np.mean(train_labels)
        # train_std = np.std(train_labels)
        # train_labels = (train_labels - train_mu) / train_std

        # import seaborn as sns
        # sns.histplot(data=train_labels, kde=True, bins=100)
        # plt.show()
        # first_label = (first_label - train_mu) / train_std
        # last_label = (last_label - train_mu) / train_std
        # plt.scatter(x=first_label, y=last_label)
        # plt.show()

        all_weights = np.ones_like(train_labels)

        if self.use_sample_weight_by_budget:
            if self.sample_weight_by_budget_strategy is not None:
                if self.sample_weight_by_budget_strategy.isdigit():
                    power_value = int(self.sample_weight_by_budget_strategy)
                    weight_fn = lambda w: np.power(w / self.max_budgets[self.fidelity_name], power_value)
                elif self.sample_weight_by_budget_strategy == "softmax":
                    weight_fn = lambda w: np.exp(w - np.max(w))
                else:
                    raise NotImplementedError
            else:
                weight_fn = lambda w: w

            train_weights = np.zeros_like(train_budgets)
            for i in np.unique(train_hp_indices):
                budgets = train_budgets[train_hp_indices == i]

                weights = budgets.astype(np.float32)
                weights = weight_fn(weights)
                weights /= weights.sum()
                weights *= weights.shape[0]
                train_weights[train_hp_indices == i] = weights
            all_weights = train_weights

        if self.use_sample_weight_by_label:
            power = 1
            if isinstance(self.use_sample_weight_by_label, int):
                power = self.use_sample_weight_by_label
            weights = train_labels.copy()
            # max_weight = np.max(weights)
            # min_weight = np.min(weights)
            # if max_weight != min_weight:
            #     weights = (weights - min_weight) / (max_weight - min_weight)
            # weights = np.abs(np.exp(-power * weights) - np.exp(-power)) / (1 - np.exp(-power))
            weights = 1 / (weights + 1e-3)
            weights *= weights.shape[0] / weights.sum()

            all_weights = weights * all_weights
            all_weights *= all_weights.shape[0] / all_weights.sum()

        train_weights = all_weights

        if self.use_y_constraint_weights:
            if isinstance(self.use_y_constraint_weights, int):
                power = self.use_y_constraint_weights
                y_constraint_weights = train_max_budget
                y_constraint_weights = np.abs(np.exp(-power * y_constraint_weights) - np.exp(-power)) / (
                    1 - np.exp(-power))
            else:
                y_constraint_weights = 1 - train_max_budget

            if self.use_sample_weight_by_budget or self.use_sample_weight_by_label:
                train_weights = np.expand_dims(train_weights, axis=1)
                y_constraint_weights = np.expand_dims(y_constraint_weights, axis=1)
                train_weights = np.concatenate([train_weights, y_constraint_weights], axis=1)
            else:
                train_weights = y_constraint_weights

        if self.use_sample_weight_by_label or self.use_y_constraint_weights or self.use_sample_weight_by_budget:
            train_weights = torch.from_numpy(train_weights)
        else:
            train_weights = None

        # This creates a copy
        train_examples = self.hp_candidates[train_hp_indices]

        train_examples = torch.from_numpy(train_examples)
        train_labels = torch.from_numpy(train_labels)
        train_budgets = torch.from_numpy(train_budgets)
        train_curves = torch.from_numpy(train_curves) if train_curves is not None else None

        # make validation dataset
        val_hp_indices = all_hp_indices[val_hp_index]
        if validation_mode == "best":
            val_labels = all_best_labels[val_hp_index]
        else:
            val_labels = all_labels[val_hp_index]
        val_budgets = all_budgets[val_hp_index]
        val_curves = all_curves[val_hp_index] if all_curves is not None else None

        # val_labels = (val_labels - train_mu) / train_std
        # sns.histplot(data=val_labels, kde=True, bins=100)
        # plt.show()

        val_examples = self.hp_candidates[val_hp_indices]

        val_examples = torch.from_numpy(val_examples)
        val_labels = torch.from_numpy(val_labels)
        val_budgets = torch.from_numpy(val_budgets)
        val_curves = torch.from_numpy(val_curves) if val_curves is not None else None

        train_dataset = TabularDataset(
            X=train_examples,
            Y=train_labels,
            budgets=train_budgets,
            curves=train_curves,
            use_sample_weights=self.use_sample_weights,
            use_sample_weight_by_budget=self.use_sample_weight_by_budget,
            weights=train_weights
        )

        val_dataset = TabularDataset(
            X=val_examples,
            Y=val_labels,
            budgets=val_budgets,
            curves=val_curves,
        )

        return train_dataset, val_dataset, target_normalization_value
