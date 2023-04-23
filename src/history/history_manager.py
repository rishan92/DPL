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
from numpy.typing import NDArray


class HistoryManager:
    def __init__(self, hp_candidates, max_benchmark_epochs, fantasize_step, use_learning_curve, use_learning_curve_mask,
                 fill_value='zero'):
        assert fill_value in ["zero", "last"], "Invalid fill value mode"
        # assert predict_mode in ["end_budget", "next_budget"], "Invalid predict mode"
        # assert curve_size_mode in ["fixed", "variable"], "Invalid curve size mode"
        self.hp_candidates = hp_candidates
        self.max_benchmark_epochs = max_benchmark_epochs
        self.fill_value = fill_value
        self.use_learning_curve = use_learning_curve
        self.use_learning_curve_mask = use_learning_curve_mask

        # the keys will be hyperparameter indices while the value
        # will be a list with all the budgets evaluated for examples
        # and with all performances for the performances
        self.examples: Dict[int, NDArray[int]] = dict()
        self.performances: Dict[int, List[float]] = dict()

        self.last_point = None

        self.fantasize_step = fantasize_step

        self.cnn_kernel_size = 3  # TODO: get this from dyhpo model hyperparameters

        self.is_history_modified = True
        self.cached_train_dataset = None

    def get_initial_empty_value(self):
        initial_empty_value = self.get_mean_initial_value() if self.fill_value == 'last' else 0
        return initial_empty_value

    def add(self, hp_index: int, b: int, hp_curve: List[float]):
        # TODO: check if arange should start with 1 or zero
        self.examples[hp_index] = np.arange(1, b + 1)
        self.performances[hp_index] = hp_curve

        initial_empty_value = self.get_initial_empty_value()
        self.last_point = (hp_index, b, hp_curve[b - 1], hp_curve[0:b - 1] if b > 1 else [initial_empty_value])

        self.is_history_modified = True

    def get_evaluated_budgets(self, suggested_hp_index):
        if suggested_hp_index in self.examples:
            return self.examples[suggested_hp_index]
        else:
            return []

    def get_last_sample(self, curve_size_mode):
        newp_index, newp_budget, newp_performance, newp_curve = self.last_point

        modified_curve = self.get_processed_curves(curves=[newp_curve], curve_size_mode=curve_size_mode,
                                                   real_budgets=newp_budget)

        new_example = torch.tensor(self.hp_candidates[newp_index], dtype=torch.float32)
        new_example = torch.unsqueeze(new_example, dim=0)
        newp_budget = torch.tensor([newp_budget], dtype=torch.float32) / self.max_benchmark_epochs
        newp_performance = torch.tensor([newp_performance], dtype=torch.float32)

        if modified_curve is not None:
            modified_curve = torch.from_numpy(modified_curve)
        else:
            modified_curve = torch.tensor([0], dtype=torch.float32)
            modified_curve = torch.unsqueeze(modified_curve, dim=0)

        last_sample = (new_example, newp_performance, newp_budget, modified_curve)
        return last_sample

    def history_configurations(self, curve_size_mode) -> \
        Tuple[NDArray[int], NDArray[np.float32], NDArray[np.float32], Optional[NDArray[np.float32]]]:
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
        initial_empty_value = self.get_initial_empty_value()

        for hp_index in self.examples:
            budgets = self.examples[hp_index]
            performances = self.performances[hp_index]

            for budget in budgets:
                train_indices.append(hp_index)
                train_budgets.append(budget)
                train_labels.append(performances[budget - 1])
                train_curve = performances[:budget - 1] if budget > 1 else [initial_empty_value]
                train_curves.append(train_curve)

        train_curves = self.get_processed_curves(curves=train_curves, curve_size_mode=curve_size_mode,
                                                 real_budgets=train_budgets)

        train_indices = np.array(train_indices, dtype=int)
        train_budgets = np.array(train_budgets, dtype=np.float32)
        train_labels = np.array(train_labels, dtype=np.float32)

        return train_indices, train_labels, train_budgets, train_curves

    def get_train_dataset(self, curve_size_mode) -> TabularDataset:
        """This method is called to prepare the necessary training dataset
        for training a model.

        Returns:
            train_dataset: A dataset consisting of examples, labels, budgets
                and learning curves.
        """
        if not self.is_history_modified:
            return self.cached_train_dataset

        hp_indices, train_labels, train_budgets, train_curves = self.history_configurations(curve_size_mode)

        # scale budgets to [0, 1]
        train_budgets = train_budgets / self.max_benchmark_epochs

        # This creates a copy
        train_examples = self.hp_candidates[hp_indices]

        train_examples = torch.from_numpy(train_examples)
        train_labels = torch.from_numpy(train_labels)
        train_budgets = torch.from_numpy(train_budgets)
        train_curves = torch.from_numpy(train_curves) if train_curves is not None else None

        train_dataset = TabularDataset(
            X=train_examples,
            Y=train_labels,
            budgets=train_budgets,
            curves=train_curves,
        )

        self.cached_train_dataset = train_dataset
        self.is_history_modified = False

        return train_dataset

    def get_predict_curves_dataset(self, hp_index, curve_size_mode):
        curves = []
        real_budgets = []
        if hp_index in self.examples:
            budgets = self.examples[hp_index]
            max_budget = max(budgets)
            performances = self.performances[hp_index]
            for budget, performance in zip(budgets, performances):
                real_budgets.append(budget)
                train_curve = performances[:budget - 1] if budget > 1 else [0.0]
                curves.append(train_curve)
        else:
            max_budget = 0
            real_budgets.append(0)
            curves.append([0])

        curves = self.get_processed_curves(curves=curves, curve_size_mode=curve_size_mode, real_budgets=real_budgets)

        p_config = self.hp_candidates[hp_index]
        p_config = torch.tensor(p_config, dtype=torch.float32)
        p_config = p_config.expand(self.max_benchmark_epochs, -1)

        real_budgets = np.arange(1, self.max_benchmark_epochs + 1)
        p_budgets = torch.tensor(real_budgets / self.max_benchmark_epochs, dtype=torch.float32)

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

        return pred_test_data, real_budgets, max_budget

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
        hp_indices, budgets, real_budgets, hp_curves = self.generate_candidate_configurations(predict_mode,
                                                                                              curve_size_mode)
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
            hp_budgets = real_budgets

        hp_indices = np.array(hp_indices, dtype=int)
        hp_budgets = np.array(hp_budgets, dtype=np.float32)
        real_budgets = np.array(real_budgets, dtype=int)

        return hp_indices, hp_budgets, real_budgets, hp_curves
