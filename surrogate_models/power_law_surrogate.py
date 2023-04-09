from copy import deepcopy
import logging
import os
import time
from typing import List, Tuple
from loguru import logger
import numpy as np
import random
from pathlib import Path
from scipy.stats import norm
import torch
from torch.utils.data import DataLoader

from data_loader.tabular_data_loader import WrappedDataLoader
from dataset.tabular_dataset import TabularDataset
from models.ensemble_model import EnsembleModel
from data_loader.surrogate_data_loader import SurrogateDataLoader


class PowerLawSurrogate:

    def __init__(
        self,
        hp_candidates: np.ndarray,
        surrogate_configs: dict = None,
        seed: int = 11,
        max_benchmark_epochs: int = 52,
        ensemble_size: int = 5,
        nr_epochs: int = 250,
        fantasize_step: int = 1,
        minimization: bool = True,
        total_budget: int = 1000,
        device: str = None,
        output_path: str = '.',
        dataset_name: str = 'unknown',
        pretrain: bool = False,
        backbone: str = 'power_law',
        max_value: float = 100,
        min_value: float = 0,
        fill_value: str = 'zero',
    ):
        """
        Args:
            hp_candidates: np.ndarray
                The full list of hyperparameter candidates for a given dataset.
            surrogate_configs: dict
                The model configurations for the surrogate.
            seed: int
                The seed that will be used for the surrogate.
            max_benchmark_epochs: int
                The maximal budget that a hyperparameter configuration
                has been evaluated in the benchmark for.
            ensemble_size: int
                The number of members in the ensemble.
            nr_epochs: int
                Number of epochs for which the surrogate should be
                trained.
            fantasize_step: int
                The number of steps for which we are looking ahead to
                evaluate the performance of a hpc.
            minimization: bool
                If for the evaluation metric, the lower the value the better.
            total_budget: int
                The total budget given. Used to calculate the initialization
                percentage.
            device: str
                The device where the experiment will be run on.
            output_path: str
                The path where all the output will be stored.
            dataset_name: str
                The name of the dataset that the experiment will be run on.
            pretrain: bool
                If the surrogate will be pretrained before with a synthetic
                curve.
            backbone: str
                The backbone, which can either be 'power_law' or 'nn'.
            max_value: float
                The maximal value for the dataset.
            min_value: float
                The minimal value for the dataset.
            fill_value: str = 'zero',
                The filling strategy for when learning curves are used.
                Either 'zero' or 'last' where last represents the last value.
        """
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.total_budget = total_budget
        self.fill_value = fill_value
        self.max_value = max_value
        self.min_value = min_value
        self.backbone = backbone

        self.pretrained_path = os.path.join(
            output_path,
            'power_law',
            f'checkpoint_{seed}.pth',
        )

        self.model_class = EnsembleModel
        self.model = None

        if device is None:
            self.dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.dev = torch.device(device)

        self.batch_size = 64
        self.refine_batch_size = 64

        self.hp_candidates = hp_candidates

        self.minimization = minimization
        self.seed = seed

        self.logger = logger

        # with what percentage configurations will be taken randomly instead of being sampled from the model
        self.fraction_random_configs = 0.1
        self.iteration_probabilities = np.random.rand(self.total_budget)

        # the keys will be hyperparameter indices while the value
        # will be a list with all the budgets evaluated for examples
        # and with all performances for the performances
        self.examples = dict()
        self.performances = dict()

        self.max_benchmark_epochs = max_benchmark_epochs
        self.ensemble_size = ensemble_size
        self.nr_epochs = nr_epochs
        self.refine_nr_epochs = 20
        self.fantasize_step = fantasize_step

        self.pretrain = pretrain

        initial_configurations_nr = 1
        conf_individual_budget = 1
        init_conf_indices = np.random.choice(self.hp_candidates.shape[0], initial_configurations_nr, replace=False)
        init_budgets = [i for i in range(1, conf_individual_budget + 1)]

        self.rand_init_conf_indices = []
        self.rand_init_budgets = []

        # basically add every config index up to a certain budget threshold for the initialization
        # we will go through both lists during the initialization
        for config_index in init_conf_indices:
            for config_budget in init_budgets:
                self.rand_init_conf_indices.append(config_index)
                self.rand_init_budgets.append(config_budget)

        self.initial_random_index = 0

        self.nr_features = self.hp_candidates.shape[1]
        self.best_value_observed = np.inf

        self.diverged_configs = set()

        # A tuple which will have the last evaluated point
        # It will be used in the refining process
        # Tuple(config_index, budget, performance, curve)
        self.last_point = None

        self.initial_full_training_trials = 10

        # a flag if the surrogate should be trained
        self.train = True

        # the times it was refined
        self.refine_counter = 0
        # the surrogate iteration counter
        self.iterations_counter = 0
        # info dict to drop every surrogate iteration
        self.info_dict = dict()

        # the start time for the overhead of every surrogate iteration
        # will be recorded here
        self.suggest_time_duration = 0

        self.output_path = output_path
        self.dataset_name = dataset_name

        self.no_improvement_threshold = int(self.max_benchmark_epochs + 0.2 * self.max_benchmark_epochs)
        self.no_improvement_patience = 0

    def _prepare_dataset(self) -> TabularDataset:
        """This method is called to prepare the necessary training dataset
        for training a model.

        Returns:
            train_dataset: A dataset consisting of examples, labels, budgets
                and learning curves.
        """
        train_examples, train_labels, train_budgets, train_curves = self.history_configurations()

        train_curves = self.prepare_training_curves(train_budgets, train_curves)
        train_examples = np.array(train_examples, dtype=np.single)
        train_labels = np.array(train_labels, dtype=np.single)
        train_budgets = np.array(train_budgets, dtype=np.single)

        # scale budgets to [0, 1]
        train_budgets = train_budgets / self.max_benchmark_epochs

        train_dataset = TabularDataset(
            train_examples,
            train_labels,
            train_budgets,
            train_curves,
        )

        return train_dataset

    def _train_surrogate(self, pretrain: bool = False, should_refine: bool = False,
                         should_weight_last_sample: bool = False):
        """Train the surrogate model.

        Trains all the models of the ensemble
        with different initializations and different
        data orders.

        Args:
            pretrain: bool
                If we have pretrained weights and we will just
                refine the models.
        """
        self.iterations_counter += 1
        self.logger.info(f'Iteration number: {self.iterations_counter}')

        train_dataset = self._prepare_dataset()

        if pretrain:
            should_refine = True,
            should_weight_last_sample = False

        last_sample = None
        if should_weight_last_sample:
            newp_index, newp_budget, newp_performance, newp_curve = self.last_point
            new_example = np.array([self.hp_candidates[newp_index]], dtype=np.single)
            newp_missing_values = self.prepare_missing_values_channel([newp_budget])
            newp_budget = np.array([newp_budget], dtype=np.single) / self.max_benchmark_epochs
            newp_performance = np.array([newp_performance], dtype=np.single)
            modified_curve = deepcopy(newp_curve)

            difference = self.max_benchmark_epochs - len(modified_curve) - 1
            if difference > 0:
                modified_curve.extend([modified_curve[-1] if self.fill_value == 'last' else 0] * difference)

            modified_curve = np.array([modified_curve], dtype=np.single)
            newp_missing_values = np.array(newp_missing_values, dtype=np.single)

            # add depth dimension to the train_curves array and missing_value_matrix
            modified_curve = np.expand_dims(modified_curve, 1)
            newp_missing_values = np.expand_dims(newp_missing_values, 1)
            modified_curve = np.concatenate((modified_curve, newp_missing_values), axis=1)

            new_example = torch.tensor(new_example, device=self.dev)
            newp_budget = torch.tensor(newp_budget, device=self.dev)
            newp_performance = torch.tensor(newp_performance, device=self.dev)
            modified_curve = torch.tensor(modified_curve, device=self.dev)

            last_sample = (new_example, newp_performance, newp_budget, modified_curve)

        if should_refine:
            nr_epochs = self.refine_nr_epochs
            batch_size = self.refine_batch_size

            # make the training dataset here
            train_dataloader = SurrogateDataLoader(
                dataset=train_dataset, batch_size=batch_size, shuffle=True, seed=self.seed, dev=self.dev,
                should_weight_last_sample=should_weight_last_sample, last_sample=last_sample,
                # drop_last=train_dataset.X.shape[0] > batch_size and train_dataset.X.shape[0] % batch_size < 2
            )
            self.model.train()
            self.model.train_loop(nr_epochs=nr_epochs, train_dataloader=train_dataloader, reset_optimizer=True)
        else:
            nr_epochs = self.nr_epochs
            batch_size = self.batch_size

            # make the training dataset here
            train_dataloader = SurrogateDataLoader(
                dataset=train_dataset, batch_size=batch_size, shuffle=True, seed=self.seed, dev=self.dev
            )

            model_config = {'seed': self.seed}
            self.model = self.model_class(
                nr_features=train_dataset.X.shape[1],
                train_dataloader=train_dataloader,
                surrogate_configs=model_config
            )
            self.model.to(self.dev)

            self.model.train()
            self.model.train_loop(nr_epochs=nr_epochs)

        return self.model

    def _predict(self) -> Tuple[np.ndarray, np.ndarray, List, np.ndarray]:
        """
        Predict the performances of the hyperparameter configurations
        as well as the standard deviations based on the ensemble.

        Returns:
            mean_predictions, std_predictions, hp_indices, real_budgets:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
                The mean predictions and the standard deviations over
                all model predictions for the given hyperparameter
                configurations with their associated indices and budgets.

        """
        configurations, hp_indices, budgets, real_budgets, hp_curves = self.generate_candidate_configurations()
        # scale budgets to [0, 1]
        budgets = np.array(budgets, dtype=np.single)
        hp_curves = self.prepare_training_curves(real_budgets, hp_curves)
        budgets = budgets / self.max_benchmark_epochs
        real_budgets = np.array(real_budgets, dtype=np.single)
        configurations = np.array(configurations, dtype=np.single)

        configurations = torch.tensor(configurations)
        configurations = configurations.to(device=self.dev)
        budgets = torch.tensor(budgets)
        budgets = budgets.to(device=self.dev)
        hp_curves = torch.tensor(hp_curves)
        hp_curves = hp_curves.to(device=self.dev)
        network_real_budgets = torch.tensor(real_budgets / self.max_benchmark_epochs)
        network_real_budgets.to(device=self.dev)

        self.model.eval()
        predictions = self.model((configurations, budgets, network_real_budgets, hp_curves))

        mean_predictions = predictions[0]
        std_predictions = predictions[1]

        return mean_predictions, std_predictions, hp_indices, real_budgets

    def suggest(self) -> Tuple[int, int]:
        """Suggest a hyperparameter configuration and a budget
        to evaluate.

        Returns:
            suggested_hp_index, budget: Tuple[int, int]
                The index of the hyperparamter configuration to be evaluated
                and the budget for what it is going to be evaluated for.
        """
        suggest_time_start = time.time()

        if self.initial_random_index < len(self.rand_init_conf_indices):
            self.logger.info(
                'Not enough configurations to build a model. \n'
                'Returning randomly sampled configuration'
            )
            suggested_hp_index = self.rand_init_conf_indices[self.initial_random_index]
            budget = self.rand_init_budgets[self.initial_random_index]
            self.initial_random_index += 1
        else:
            mean_predictions, std_predictions, hp_indices, real_budgets = self._predict()
            best_prediction_index = self.find_suggested_config(
                mean_predictions,
                std_predictions,
            )
            # actually do the mapping between the configuration indices and the best prediction index
            suggested_hp_index = hp_indices[best_prediction_index]

            if suggested_hp_index in self.examples:
                evaluated_budgets = self.examples[suggested_hp_index]
                max_budget = max(evaluated_budgets)
                budget = max_budget + self.fantasize_step
                if budget > self.max_benchmark_epochs:
                    budget = self.max_benchmark_epochs
            else:
                budget = self.fantasize_step

        suggest_time_end = time.time()
        self.suggest_time_duration = suggest_time_end - suggest_time_start

        return suggested_hp_index, budget

    def observe(
        self,
        hp_index: int,
        b: int,
        hp_curve: List[float],
    ):
        """Receive information regarding the performance of a hyperparameter
        configuration that was suggested.

        Args:
            hp_index: int
                The index of the evaluated hyperparameter configuration.
            b: int
                The budget for which the hyperparameter configuration was evaluated.
            hp_curve: List
                The performance of the hyperparameter configuration.
        """
        for index, curve_element in enumerate(hp_curve):
            if np.isnan(curve_element):
                self.diverged_configs.add(hp_index)
                # only use the non-nan part of the curve and the corresponding
                # budget to still have the information in the network
                hp_curve = hp_curve[0:index + 1]
                b = index
                break

        if not self.minimization:
            hp_curve = np.subtract([self.max_value] * len(hp_curve), hp_curve)
            hp_curve = hp_curve.tolist()

        best_curve_value = min(hp_curve)

        self.examples[hp_index] = np.arange(1, b + 1)
        self.performances[hp_index] = hp_curve

        if self.best_value_observed > best_curve_value:
            self.best_value_observed = best_curve_value
            self.no_improvement_patience = 0
            self.logger.info(f'New Incumbent value found '
                             f'{1 - best_curve_value if not self.minimization else best_curve_value}')
        else:
            self.no_improvement_patience += 1
            if self.no_improvement_patience == self.no_improvement_threshold:
                self.train = True
                self.no_improvement_patience = 0
                self.logger.info(
                    'No improvement in the incumbent value threshold reached, '
                    'restarting training from scratch'
                )

        initial_empty_value = self.get_mean_initial_value() if self.fill_value == 'last' else 0
        if self.initial_random_index >= len(self.rand_init_conf_indices):
            performance = self.performances[hp_index]
            self.last_point = (
                hp_index, b, performance[b - 1], performance[0:b - 1] if b > 1 else [initial_empty_value])

            if self.train:
                # delete the previously stored models
                self.models = []
                if self.pretrain:
                    # TODO Load the pregiven weights.
                    pass

                self._train_surrogate(pretrain=self.pretrain)

                if self.iterations_counter <= self.initial_full_training_trials:
                    self.train = True
                else:
                    self.train = False
            else:
                self.refine_counter += 1
                self._train_surrogate(should_refine=True, should_weight_last_sample=True)

    def prepare_examples(self, hp_indices: List) -> List:
        """
        Prepare the examples to be given to the surrogate model.

        Args:
            hp_indices: List
                The list of hp indices that are already evaluated.

        Returns:
            examples: List
                A list of the hyperparameter configurations.
        """
        examples = []
        for hp_index in hp_indices:
            examples.append(self.hp_candidates[hp_index])

        return examples

    def generate_candidate_configurations(self) -> Tuple[List, List, List, List, List]:
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
                real_budgets.append(max_budget)
                learning_curve = self.performances[hp_index]

                hp_curve = learning_curve[0:max_budget - 1] if max_budget > 1 else [initial_empty_value]
            else:
                real_budgets.append(1)
                hp_curve = [initial_empty_value]

            hp_indices.append(hp_index)
            hp_budgets.append(self.max_benchmark_epochs)
            hp_curves.append(hp_curve)

        configurations = self.prepare_examples(hp_indices)

        return configurations, hp_indices, hp_budgets, real_budgets, hp_curves

    def history_configurations(self) -> Tuple[List, List, List, List]:
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
        initial_empty_value = self.get_mean_initial_value() if self.fill_value == 'last' else 0

        for hp_index in self.examples:
            budgets = self.examples[hp_index]
            performances = self.performances[hp_index]
            example = self.hp_candidates[hp_index]

            for budget in budgets:
                example_curve = performances[0:budget - 1]
                train_examples.append(example)
                train_budgets.append(budget)
                train_labels.append(performances[budget - 1])
                train_curves.append(example_curve if len(example_curve) > 0 else [initial_empty_value])

        return train_examples, train_labels, train_budgets, train_curves

    @staticmethod
    def acq(
        best_values: np.ndarray,
        mean_predictions: np.ndarray,
        std_predictions: np.ndarray,
        explore_factor: float = 0.25,
        acq_choice: str = 'ei',
    ) -> np.ndarray:
        """
        Calculate the acquisition function based on the network predictions.

        Args:
        -----
        best_values: np.ndarray
            An array with the best value for every configuration.
            Depending on the implementation it can be different for every
            configuration.
        mean_predictions: np.ndarray
            The mean values of the model predictions.
        std_predictions: np.ndarray
            The standard deviation values of the model predictions.
        explore_factor: float
            The explore factor, when ucb is used as an acquisition
            function.
        acq_choice: str
            The choice for the acquisition function to use.

        Returns
        -------
        acq_values: np.ndarray
            The values of the acquisition function for every configuration.
        """
        if acq_choice == 'ei':
            z = (np.subtract(best_values, mean_predictions))
            difference = deepcopy(z)
            not_zero_std_indicator = [False if example_std == 0.0 else True for example_std in std_predictions]
            zero_std_indicator = np.invert(not_zero_std_indicator)
            z = np.divide(z, std_predictions, where=not_zero_std_indicator)
            np.place(z, zero_std_indicator, 0)
            acq_values = np.add(np.multiply(difference, norm.cdf(z)), np.multiply(std_predictions, norm.pdf(z)))
        elif acq_choice == 'ucb':
            # we are working with error rates so we multiply the mean with -1
            acq_values = np.add(-1 * mean_predictions, explore_factor * std_predictions)
        elif acq_choice == 'thompson':
            acq_values = np.random.normal(mean_predictions, std_predictions)
        else:
            acq_values = mean_predictions

        return acq_values

    def find_suggested_config(
        self,
        mean_predictions: np.ndarray,
        mean_stds: np.ndarray,
    ) -> int:
        """Return the hyperparameter with the highest acq function value.

        Given the mean predictions and mean standard deviations from the DPL
        ensemble for every hyperparameter configuraiton, return the hyperparameter
        configuration that has the highest acquisition function value.

        Args:
            mean_predictions: np.ndarray
                The mean predictions of the ensemble for every hyperparameter
                configuration.
            mean_stds: np.ndarray
                The standard deviation predictions of the ensemble for every
                hyperparameter configuration.

        Returns:
            max_value_index: int
                the index of the maximal value.

        """
        best_values = np.array([self.best_value_observed] * mean_predictions.shape[0])
        acq_func_values = self.acq(
            best_values,
            mean_predictions,
            mean_stds,
            acq_choice='ei',
        )

        max_value_index = np.argmax(acq_func_values)

        return max_value_index

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

    def patch_curves_to_same_length(self, curves: List):
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
        self.patch_curves_to_same_length(train_curves)
        train_curves = np.array(train_curves, dtype=np.single)
        missing_value_matrix = np.array(missing_value_matrix, dtype=np.single)

        # add depth dimension to the train_curves array and missing_value_matrix
        train_curves = np.expand_dims(train_curves, 1)
        missing_value_matrix = np.expand_dims(missing_value_matrix, 1)
        train_curves = np.concatenate((train_curves, missing_value_matrix), axis=1)

        return train_curves
