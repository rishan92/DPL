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
from src.dataset.tabular_dataset import TabularDataset
import functools
from functools import partial
from numpy.typing import NDArray
from src.utils.utils import get_class_from_package, get_class_from_packages, numpy_to_torch_apply
import src.models.activation_functions
from src.benchmarks.base_benchmark import BaseBenchmark
import matplotlib.pyplot as plt


class HistoryManager:
    def __init__(self, hp_candidates, max_benchmark_epochs, fantasize_step, use_learning_curve, use_learning_curve_mask,
                 fill_value='zero', use_target_normalization=False, use_scaled_budgets=True,
                 model_output_normalization=None, cnn_kernel_size=0, target_normalization_range=None,
                 use_sample_weights=False, use_sample_weight_by_budget=False, sample_weight_by_budget_strategy=None):
        assert fill_value in ["zero", "last"], "Invalid fill value mode"
        # assert predict_mode in ["end_budget", "next_budget"], "Invalid predict mode"
        # assert curve_size_mode in ["fixed", "variable"], "Invalid curve size mode"
        self.hp_candidates = hp_candidates
        self.max_benchmark_epochs = max_benchmark_epochs
        self.fill_value = fill_value
        self.use_learning_curve = use_learning_curve
        self.use_learning_curve_mask = use_learning_curve_mask
        self.use_scaled_budgets = use_scaled_budgets
        self.cnn_kernel_size = cnn_kernel_size
        self.use_sample_weights = use_sample_weights
        self.use_sample_weight_by_budget = use_sample_weight_by_budget
        self.sample_weight_by_budget_strategy = sample_weight_by_budget_strategy

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
        self.examples: Dict[int, NDArray[int]] = dict()
        self.performances: Dict[int, List[float]] = dict()

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

    def add(self, hp_index: int, b: int, hp_curve: List[float]):
        self.examples[hp_index] = np.arange(1, b + 1)
        self.performances[hp_index] = hp_curve

        initial_empty_value = self.get_initial_empty_value()

        self.last_point = (hp_index, b, hp_curve[b - 1], hp_curve[0:b - 1] if b > 1 else [initial_empty_value])

        max_curve = np.max(hp_curve)
        self.max_curve_value = max(self.max_curve_value, max_curve)
        # self.max_curve_value = 10

        self.is_train_data_modified = True
        self.is_test_data_modified = True

    def get_evaluated_budgets(self, suggested_hp_index):
        if suggested_hp_index in self.examples:
            return self.examples[suggested_hp_index]
        else:
            return []

    def get_evaluted_indices(self):
        if len(self.examples) == 0:
            return []
        else:
            return list(self.examples.keys())

    def get_last_sample(self, curve_size_mode):
        newp_index, newp_budget, newp_performance, newp_curve = self.last_point

        modified_curve = self.get_processed_curves(curves=[newp_curve], curve_size_mode=curve_size_mode,
                                                   real_budgets=newp_budget)

        new_example = torch.tensor(self.hp_candidates[newp_index], dtype=torch.float32)
        new_example = torch.unsqueeze(new_example, dim=0)

        newp_budget = torch.tensor([newp_budget], dtype=torch.float32)
        if self.use_scaled_budgets:
            newp_budget = newp_budget / self.max_benchmark_epochs

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

        if self.cached_train_dataset is not None:
            weight = self.cached_train_dataset.get_weight(x=new_example, budget=newp_budget)
        else:
            weight = torch.tensor([1.0])

        last_sample = (new_example, newp_performance, newp_budget, modified_curve, weight)
        return last_sample

    def history_configurations(self, curve_size_mode) -> \
        Tuple[NDArray[int], NDArray[np.float32], NDArray[np.float32],
              Optional[NDArray[np.float32]], NDArray[np.float32]]:
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
        initial_empty_value = self.get_initial_empty_value()

        if self.sample_weight_by_budget_strategy is not None:
            if self.sample_weight_by_budget_strategy.isdigit():
                power_value = int(self.sample_weight_by_budget_strategy)
                weight_fn = lambda w: np.power(w / self.max_benchmark_epochs, power_value)
            elif self.sample_weight_by_budget_strategy == "softmax":
                weight_fn = lambda w: np.exp(w - np.max(w))
            else:
                raise NotImplementedError
        else:
            weight_fn = lambda w: w

        for hp_index in self.examples:
            budgets = self.examples[hp_index]
            performances = self.performances[hp_index]

            weights = budgets.astype(np.float32)
            weights = weight_fn(weights)
            weights /= weights.sum()
            weights *= weights.shape[0]

            for i, budget in enumerate(budgets):
                train_indices.append(hp_index)
                train_budgets.append(budget)
                train_labels.append(performances[budget - 1])
                if self.use_learning_curve:
                    train_curve = performances[:budget - 1] if budget > 1 else [initial_empty_value]
                    train_curves.append(train_curve)
                train_weights.append(weights[i])

        train_curves = self.get_processed_curves(curves=train_curves, curve_size_mode=curve_size_mode,
                                                 real_budgets=train_budgets)

        train_indices = np.array(train_indices, dtype=int)
        train_budgets = np.array(train_budgets, dtype=np.float32)
        train_labels = np.array(train_labels, dtype=np.float32)
        train_weights = np.array(train_weights, dtype=np.float32)

        return train_indices, train_labels, train_budgets, train_curves, train_weights

    def get_train_dataset(self, curve_size_mode) -> TabularDataset:
        """This method is called to prepare the necessary training dataset
        for training a model.

        Returns:
            train_dataset: A dataset consisting of examples, labels, budgets
                and learning curves.
        """
        if not self.is_train_data_modified:
            return self.cached_train_dataset

        hp_indices, train_labels, train_budgets, train_curves, train_weights = \
            self.history_configurations(curve_size_mode)

        if self.use_scaled_budgets:
            # scale budgets to [0, 1]
            train_budgets = train_budgets / self.max_benchmark_epochs

        if self.use_target_normalization:
            train_labels = self.target_normalization_fn(train_labels)

        if self.model_output_normalization_fn:
            train_labels = self.model_output_normalization_fn(train_labels)

        # This creates a copy
        train_examples = self.hp_candidates[hp_indices]

        train_examples = torch.from_numpy(train_examples)
        train_labels = torch.from_numpy(train_labels)
        train_budgets = torch.from_numpy(train_budgets)
        train_curves = torch.from_numpy(train_curves) if train_curves is not None else None
        train_weights = torch.from_numpy(train_weights) if self.use_sample_weight_by_budget else None

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

    def get_predict_curves_dataset(self, hp_index, curve_size_mode):
        curves = []
        real_budgets = []
        if hp_index in self.examples:
            budgets = self.examples[hp_index]
            max_train_budget = max(budgets)
            performances = self.performances[hp_index]
            for budget, performance in zip(budgets, performances):
                real_budgets.append(budget)
                train_curve = performances[:budget - 1] if budget > 1 else [0.0]
                curves.append(train_curve)
        else:
            max_train_budget = 0
            real_budgets.append(0)
            curves.append([0])

        curves = self.get_processed_curves(curves=curves, curve_size_mode=curve_size_mode, real_budgets=real_budgets)

        p_config = self.hp_candidates[hp_index]
        p_config = torch.tensor(p_config, dtype=torch.float32)
        p_config = p_config.expand(self.max_benchmark_epochs, -1)

        real_budgets = np.arange(1, self.max_benchmark_epochs + 1)

        p_budgets = torch.tensor(real_budgets, dtype=torch.float32)
        if self.use_scaled_budgets:
            p_budgets = p_budgets / self.max_benchmark_epochs

        p_curve = None
        if curves is not None:
            p_curve = torch.tensor(curves, dtype=torch.float32)
            p_curve_last_row = p_curve[-1].unsqueeze(0)
            p_curve_num_repeats = self.max_benchmark_epochs - p_curve.size(0)
            repeated_last_row = p_curve_last_row.repeat_interleave(p_curve_num_repeats, dim=0)
            p_curve = torch.cat((p_curve, repeated_last_row), dim=0)

        pred_test_data = TabularDataset(
            X=p_config,
            budgets=p_budgets,
            curves=p_curve
        )

        return pred_test_data, real_budgets, max_train_budget

    def get_processed_curves(self, curves, curve_size_mode, real_budgets) -> Optional[NDArray[np.float32]]:
        if self.use_learning_curve:
            if curve_size_mode == "variable":
                min_size = self.cnn_kernel_size
            elif curve_size_mode == "fixed":
                min_size = self.max_benchmark_epochs - 1
            else:
                raise NotImplementedError

            curves = self.patch_curves_to_same_length(curves=curves, min_size=min_size)

            if self.use_learning_curve_mask:
                curves = self.add_curve_missing_value_mask(curves, real_budgets)
        else:
            curves = None
        return curves

    def patch_curves_to_same_length(self, curves: List[List[float]], min_size: int) -> NDArray[np.float32]:
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
                performance = self.performances[example_index][fidelity - 1]
            except IndexError:
                performance = self.performances[example_index][-1]
            config_values.append(performance)

        # lowest error corresponds to best value
        best_value = min(config_values)

        return best_value

    def calculate_fidelity_ymax_dyhpo(self, fidelity: int):
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
            try:
                performance = self.performances[example_index][fidelity - 1]
                exact_fidelity_config_values.append(performance)
            except IndexError:
                learning_curve = self.performances[example_index]
                # The hyperparameter was not evaluated until fidelity, or more.
                # Take the maximum value from the curve.
                lower_fidelity_config_values.append(min(learning_curve))

        if len(exact_fidelity_config_values) > 0:
            # lowest error corresponds to best value
            best_value = min(exact_fidelity_config_values)
        else:
            best_value = min(lower_fidelity_config_values)

        return best_value

    def add_curve_missing_value_mask(self, curves: NDArray[np.float32], budgets: List[int]) -> NDArray[np.float32]:
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

    def prepare_missing_values_masks(self, budgets: List[int], size: int) -> NDArray[bool]:
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
        for performance_curve in self.performances.values():
            first_values.append(performance_curve[0])

        mean_initial_value = np.mean(first_values)

        return mean_initial_value

    def get_candidate_configurations_dataset(self, predict_mode, curve_size_mode) -> \
        Tuple[TabularDataset, NDArray[int], NDArray[int]]:

        if not self.is_test_data_modified:
            return self.cached_test_dataset

        hp_indices, budgets, real_budgets, hp_curves = self.generate_candidate_configurations(predict_mode,
                                                                                              curve_size_mode)

        if self.use_scaled_budgets:
            # scale budgets to [0, 1]
            budgets = budgets / self.max_benchmark_epochs

        # This creates a copy
        configurations = self.hp_candidates[hp_indices]

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
        Tuple[NDArray[int], NDArray[np.float32], NDArray[int], Optional[NDArray[np.float32]]]:
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
        initial_empty_value = self.get_mean_initial_value() if self.fill_value == 'last' else 0

        for hp_index in range(0, self.hp_candidates.shape[0]):

            if hp_index in self.examples:
                budgets = self.examples[hp_index]
                # Take the max budget evaluated for a certain hpc
                max_budget = budgets[-1]
                if max_budget == self.max_benchmark_epochs:
                    continue
                real_budgets.append(max_budget + self.fantasize_step)
                learning_curve = self.performances[hp_index]

                hp_curve = learning_curve[:max_budget - 1] if max_budget > 1 else [initial_empty_value]
            else:
                real_budgets.append(self.fantasize_step)
                hp_curve = [initial_empty_value]

            hp_indices.append(hp_index)
            hp_budgets.append(self.max_benchmark_epochs)
            hp_curves.append(hp_curve)

        hp_curves = self.get_processed_curves(curves=hp_curves, curve_size_mode=curve_size_mode,
                                              real_budgets=hp_budgets)

        if predict_mode == "next_budget":
            # make sure there is a copy happening because hp_budgets get normalized and real_budgets does not.
            # Creating np.array below copies the data.
            hp_budgets = real_budgets

        hp_indices = np.array(hp_indices, dtype=int)
        hp_budgets = np.array(hp_budgets, dtype=np.float32)
        real_budgets = np.array(real_budgets, dtype=int)

        return hp_indices, hp_budgets, real_budgets, hp_curves

    def all_configurations(self, curve_size_mode, benchmark: BaseBenchmark) -> \
        Tuple[NDArray[int], NDArray[np.float32], NDArray[np.float32], Optional[NDArray[np.float32]], NDArray[bool]]:

        train_indices = []
        train_labels = []
        train_budgets = []
        train_curves = []
        is_up_curve = []
        initial_empty_value = self.get_initial_empty_value()

        budgets = range(1, self.max_benchmark_epochs + 1)
        for hp_index in range(0, self.hp_candidates.shape[0]):
            performances = benchmark.get_curve(hp_index, budget=self.max_benchmark_epochs)
            start_performance = performances[0]
            for budget in budgets:
                train_indices.append(hp_index)
                train_budgets.append(budget)
                train_labels.append(performances[budget - 1])
                is_up_curve.append(performances[budget - 1] > start_performance)
                if self.use_learning_curve:
                    train_curve = performances[:budget - 1] if budget > 1 else [initial_empty_value]
                    train_curves.append(train_curve)

        train_curves = self.get_processed_curves(curves=train_curves, curve_size_mode=curve_size_mode,
                                                 real_budgets=train_budgets)

        train_indices = np.array(train_indices, dtype=int)
        train_budgets = np.array(train_budgets, dtype=np.float32)
        train_labels = np.array(train_labels, dtype=np.float32)
        is_up_curve = np.array(is_up_curve, dtype=bool)

        return train_indices, train_labels, train_budgets, train_curves, is_up_curve

    def get_check_train_validation_dataset(self, curve_size_mode, benchmark: BaseBenchmark,
                                           validation_configuration_ratio, validation_curve_ratio, validation_mode,
                                           seed) -> (
        TabularDataset, TabularDataset):
        """This method is called to prepare the necessary training dataset
        for training a model.

        Returns:
            train_dataset: A dataset consisting of examples, labels, budgets
                and learning curves.
        """

        all_hp_indices, all_labels, all_budgets, all_curves, all_is_up_curve = \
            self.all_configurations(curve_size_mode, benchmark=benchmark)

        self.max_curve_value = np.max(all_labels)
        target_normalization_value = self.set_target_normalization_value()

        indices = np.arange(self.hp_candidates.shape[0])
        np.random.seed(seed)
        val_indices = np.random.choice(indices,
                                       size=int(self.hp_candidates.shape[0] * validation_configuration_ratio) + 1,
                                       replace=False)
        if len(val_indices) == self.hp_candidates.shape[0]:
            val_indices = []

        budgets = np.arange(1, self.max_benchmark_epochs + 1)
        val_budgets_indices = budgets[int(self.max_benchmark_epochs * (1 - validation_curve_ratio)) + 1:]

        val_hp_index = np.isin(all_hp_indices, val_indices)
        val_budget_index = np.isin(all_budgets, val_budgets_indices)
        val_hp_index = np.logical_or(val_hp_index, val_budget_index)

        train_hp_index = ~val_hp_index

        if validation_mode == "end":
            val_budget_index = all_budgets == self.max_benchmark_epochs
            val_hp_index = np.logical_and(val_hp_index, val_budget_index)

        # a = val_hp_index.sum()
        # b = all_is_up_curve.sum()
        # filter out the curve points that goes up from the validation dataset
        all_is_down_curve = ~all_is_up_curve
        val_hp_index = np.logical_and(val_hp_index, all_is_down_curve)
        # c = val_hp_index.sum()

        if self.use_scaled_budgets:
            # scale budgets to [0, 1]
            all_budgets = all_budgets / self.max_benchmark_epochs

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

        # This creates a copy
        train_examples = self.hp_candidates[train_hp_indices]

        train_examples = torch.from_numpy(train_examples)
        train_labels = torch.from_numpy(train_labels)
        train_budgets = torch.from_numpy(train_budgets)
        train_curves = torch.from_numpy(train_curves) if train_curves is not None else None

        # make validation dataset
        val_hp_indices = all_hp_indices[val_hp_index]
        val_labels = all_labels[val_hp_index]
        val_budgets = all_budgets[val_hp_index]
        val_curves = all_curves[val_hp_index] if all_curves is not None else None

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
            use_sample_weight_by_budget=self.use_sample_weight_by_budget
        )

        val_dataset = TabularDataset(
            X=val_examples,
            Y=val_labels,
            budgets=val_budgets,
            curves=val_curves,
        )

        return train_dataset, val_dataset, target_normalization_value
