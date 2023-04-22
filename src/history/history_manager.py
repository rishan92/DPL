from copy import deepcopy
import os
import time
from typing import List, Tuple, Dict, Optional, Any
from loguru import logger
import numpy as np
import random
from scipy.stats import norm
import torch
from types import SimpleNamespace
import global_variables as gv
from src.dataset.tabular_dataset import TabularDataset


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
        self.examples = dict()
        self.performances = dict()

        self.last_point = None

        self.fantasize_step = fantasize_step

        self.cnn_kernel_size = 3  # TODO: get this from dyhpo model hyperparameters

        self.is_history_modified = True
        self.cached_train_dataset = None

    def get_initial_empty_value(self):
        initial_empty_value = self.get_mean_initial_value() if self.fill_value == 'last' else 0
        return initial_empty_value

    def add(self, hp_index, b, hp_curve):
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

    def get_last_sample(self):
        newp_index, newp_budget, newp_performance, newp_curve = self.last_point

        new_example = torch.tensor(self.hp_candidates[newp_index])
        new_example = torch.unsqueeze(new_example, dim=0)
        newp_budget = torch.tensor([newp_budget]) / self.max_benchmark_epochs
        newp_performance = torch.tensor([newp_performance])

        if self.use_learning_curve:
            modified_curve = deepcopy(newp_curve)

            difference = self.max_benchmark_epochs - len(modified_curve) - 1
            if difference > 0:
                modified_curve.extend([modified_curve[-1] if self.fill_value == 'last' else 0] * difference)

            modified_curve = np.array([modified_curve], dtype=np.single)

            newp_missing_values = self.prepare_missing_values_channel([newp_budget])
            newp_missing_values = np.array(newp_missing_values, dtype=np.single)

            # add depth dimension to the train_curves array and missing_value_matrix
            modified_curve = np.expand_dims(modified_curve, 1)
            newp_missing_values = np.expand_dims(newp_missing_values, 1)
            modified_curve = np.concatenate((modified_curve, newp_missing_values), axis=1)
            modified_curve = torch.tensor(modified_curve)
        else:
            modified_curve = torch.tensor([0])
            modified_curve = torch.unsqueeze(modified_curve, dim=1)

        last_sample = (new_example, newp_performance, newp_budget, modified_curve)
        return last_sample

    def history_configurations(self, curve_size_mode) -> Tuple[List, List, List, List]:
        """
        Generate the configurations, labels, budgets and curves
        based on the history of evaluated configurations.

        Returns:
            (train_examples, train_labels, train_budgets, train_curves): Tuple
                A tuple of examples, labels and budgets for the
                configurations evaluated so far.
        """
        train_examples = []
        train_labels = []
        train_budgets = []
        train_curves = []
        initial_empty_value = self.get_initial_empty_value()

        for hp_index in self.examples:
            budgets = self.examples[hp_index]
            performances = self.performances[hp_index]
            example = self.hp_candidates[hp_index]

            for budget in budgets:
                train_examples.append(example)
                train_budgets.append(budget)
                train_labels.append(performances[budget - 1])
                train_curve = performances[:budget - 1] if budget > 1 else [initial_empty_value]
                train_curves.append(train_curve)

        if self.use_learning_curve:
            if curve_size_mode == "variable":
                train_curves = self.patch_curves_to_same_variable_length(curves=train_curves,
                                                                         min_size=self.cnn_kernel_size)
            elif curve_size_mode == "fixed":
                train_curves = self.prepare_training_curves(train_budgets, train_curves)
            else:
                raise NotImplementedError
        else:
            train_curves = None

        return train_examples, train_labels, train_budgets, train_curves

    def prepare_dataset(self, curve_size_mode):  # -> TabularDataset:
        """This method is called to prepare the necessary training dataset
        for training a model.

        Returns:
            train_dataset: A dataset consisting of examples, labels, budgets
                and learning curves.
        """
        if not self.is_history_modified:
            return self.cached_train_dataset

        train_examples, train_labels, train_budgets, train_curves = self.history_configurations(curve_size_mode)

        train_examples = np.array(train_examples, dtype=np.single)
        train_labels = np.array(train_labels, dtype=np.single)
        train_budgets = np.array(train_budgets, dtype=np.single)

        # scale budgets to [0, 1]
        train_budgets = train_budgets / self.max_benchmark_epochs

        train_examples = torch.tensor(train_examples)
        train_labels = torch.tensor(train_labels)
        train_budgets = torch.tensor(train_budgets)
        train_curves = torch.tensor(train_curves) if train_curves else None

        train_dataset = TabularDataset(
            X=train_examples,
            Y=train_labels,
            budgets=train_budgets,
            curves=train_curves,
        )

        self.cached_train_dataset = train_dataset
        self.is_history_modified = False

        return train_dataset

    def get_curves(self, hp_index, curve_size_mode):
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

        if self.use_learning_curve:
            if curve_size_mode == "variable":
                curves = self.patch_curves_to_same_variable_length(curves=curves, min_size=self.cnn_kernel_size)
            elif curve_size_mode == "fixed":
                curves = self.prepare_training_curves(real_budgets, curves)
            else:
                raise NotImplementedError
        else:
            curves = None

        return curves, max_budget

    @staticmethod
    def patch_curves_to_same_variable_length(curves, min_size):
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
        max_curve_length = 0
        for curve in curves:
            if len(curve) > max_curve_length:
                max_curve_length = len(curve)

        max_curve_length = max(max_curve_length, min_size)

        for curve in curves:
            difference = max_curve_length - len(curve)
            if difference > 0:
                curve.extend([0.0] * difference)

        return curves

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
                lower_fidelity_config_values.append(max(learning_curve))

        if len(exact_fidelity_config_values) > 0:
            # lowest error corresponds to best value
            best_value = max(exact_fidelity_config_values)
        else:
            best_value = max(lower_fidelity_config_values)

        return best_value

    def patch_curves_to_same_fixed_length(self, curves: List):
        """
        Patch the given curves to the same length.

        Finds the maximum curve length and patches all
        other curves that are shorter with zeroes.

        Args:
            curves: List
                The hyperparameter curves.
        """
        for curve in curves:
            difference = self.max_benchmark_epochs - len(curve) - 1
            if difference > 0:
                fill_value = [curve[-1]] if self.fill_value == 'last' else [0]
                curve.extend(fill_value * difference)

    def prepare_missing_values_channel(self, budgets: List) -> List:
        """Prepare an additional channel for learning curves.

        The additional channel will represent an existing learning
        curve value with a 1 and a missing learning curve value with
        a 0.

        Args:
            budgets: List
                A list of budgets for every training point.

        Returns:
            missing_value_curves: List
                A list of curves representing existing or missing
                values for the training curves of the training points.
        """
        missing_value_curves = []

        for i in range(len(budgets)):
            budget = budgets[i]
            budget = budget - 1
            budget = int(budget)

            if budget > 0:
                example_curve = [1] * budget
            else:
                example_curve = []

            difference_in_curve = self.max_benchmark_epochs - len(example_curve) - 1
            if difference_in_curve > 0:
                example_curve.extend([0] * difference_in_curve)
            missing_value_curves.append(example_curve)

        return missing_value_curves

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

    def prepare_training_curves(
        self,
        train_budgets: List[int],
        train_curves: List[float]
    ) -> np.ndarray:
        """Prepare the configuration performance curves for training.

        For every configuration training curve, add an extra dimension
        regarding the missing values, as well as extend the curve to have
        a fixed uniform length for all.

        Args:
            train_budgets: List
                A list of the budgets for all training points.
            train_curves: List
                A list of curves that pertain to every training point.

        Returns:
            train_curves: np.ndarray
                The transformed training curves.
        """
        missing_value_matrix = self.prepare_missing_values_channel(train_budgets)
        self.patch_curves_to_same_fixed_length(train_curves)
        train_curves = np.array(train_curves, dtype=np.single)
        missing_value_matrix = np.array(missing_value_matrix, dtype=np.single)

        # add depth dimension to the train_curves array and missing_value_matrix
        train_curves = np.expand_dims(train_curves, 1)
        missing_value_matrix = np.expand_dims(missing_value_matrix, 1)
        train_curves = np.concatenate((train_curves, missing_value_matrix), axis=1)

        return train_curves

    # TODO: break this function to only handle candidates in history and make config manager handle configs
    #  not in history
    def generate_candidate_configurations(self, predict_mode, curve_size_mode) -> Tuple[List, List, List, List]:
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

        if self.use_learning_curve:
            if curve_size_mode == "variable":
                hp_curves = self.patch_curves_to_same_variable_length(curves=hp_curves, min_size=self.cnn_kernel_size)
            elif curve_size_mode == "fixed":
                hp_curves = self.prepare_training_curves(real_budgets, hp_curves)
            else:
                raise NotImplementedError(f"curve_size_mode {curve_size_mode}")
        else:
            hp_curves = None

        if predict_mode == "next_budget":
            hp_budgets = real_budgets

        return hp_indices, hp_budgets, real_budgets, hp_curves
