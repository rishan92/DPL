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
import wandb
import functools
import matplotlib.pyplot as plt
import seaborn as sns

from src.dataset.tabular_dataset import TabularDataset
from src.models.power_law.ensemble_model import EnsembleModel
from src.data_loader.surrogate_data_loader import SurrogateDataLoader
from src.models.deep_kernel_learning.dyhpo_model import DyHPOModel
import global_variables as gv
from src.history.history_manager import HistoryManager
from src.plot.wandb_logger import WANDBLogger


class PowerLawSurrogate:
    model_types = {
        'power_law': EnsembleModel,
        'dyhpo': DyHPOModel,
    }
    meta = None

    def __init__(
        self,
        hp_candidates: np.ndarray,
        surrogate_name: str = 'power_law',
        seed: int = 11,
        max_benchmark_epochs: int = 52,
        ensemble_size: int = 5,
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

        self.model_type = surrogate_name
        self.model_class = PowerLawSurrogate.model_types[surrogate_name]
        self.model = None

        assert PowerLawSurrogate.meta is not None, "Meta parameters are not set"
        self.hp = PowerLawSurrogate.meta

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

        if device is None:
            self.dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.dev = torch.device(device)

        self.hp_candidates = hp_candidates

        self.minimization = minimization
        self.seed = seed

        self.logger = logger

        # with what percentage configurations will be taken randomly instead of being sampled from the model
        self.fraction_random_configs = self.hp.fraction_random_configs
        # self.iteration_probabilities = np.random.rand(self.total_budget)

        self.max_benchmark_epochs = max_benchmark_epochs
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
        if gv.IS_DYHPO:
            self.best_value_observed = np.NINF
        else:
            self.best_value_observed = np.inf

        self.diverged_configs = set()

        # A tuple which will have the last evaluated point
        # It will be used in the refining process
        # Tuple(config_index, budget, performance, curve)
        self.last_point = None

        # the number of initial points for which we will retrain fully from scratch
        # This is basically equal to the dimensionality of the search space + 1.
        self.initial_full_training_trials = self.hp.initial_full_training_trials

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

        self.checkpoint_path = os.path.join(
            output_path,
            'checkpoints',
            f'{dataset_name}',
            f'{self.seed}',
        )

        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.history_manager = HistoryManager(
            hp_candidates=self.hp_candidates,
            max_benchmark_epochs=max_benchmark_epochs,
            fill_value=self.fill_value,
            fantasize_step=self.fantasize_step,  # TODO: remove this dependency
        )

    def get_meta(self):
        return vars(self.hp)

    @staticmethod
    def get_default_meta(model_class):
        if model_class == EnsembleModel:
            hp = {
                'fraction_random_configs': 0.1,
                'initial_full_training_trials': 10,
                'predict_mode': 'end_budget',
                'curve_size_mode': 'fixed',
            }
        elif model_class == DyHPOModel:
            hp = {
                'fraction_random_configs': 0.1,
                'initial_full_training_trials': 10,
                'predict_mode': 'next_budget',
                'curve_size_mode': 'variable',
            }
        else:
            raise NotImplementedError(f"{model_class=}")
        return hp

    @classmethod
    def set_meta(cls, surrogate_name, config=None):
        config = {} if config is None else config
        model_class = cls.model_types[surrogate_name]
        default_meta = cls.get_default_meta(model_class)
        meta = {**default_meta, **config}
        model_config = model_class.set_meta(config.get("model", None))
        meta['model'] = model_config
        cls.meta = SimpleNamespace(**meta)
        return meta

    def _train_surrogate(self, pretrain: bool = False, should_refine: bool = False, load_checkpoint: bool = False):
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

        train_dataset = self.history_manager.prepare_dataset(curve_size_mode=self.meta.curve_size_mode)

        if pretrain:
            should_refine = True,
            should_weight_last_sample = False

        last_sample = self.history_manager.get_last_sample()

        if should_refine:
            self.model.train_loop(train_dataset=train_dataset, should_refine=should_refine, reset_optimizer=True,
                                  last_sample=last_sample)
        else:
            self.model = self.model_class(
                nr_features=train_dataset.X.shape[1],
                checkpoint_path=self.checkpoint_path,
                seed=self.seed
            )
            self.model.to(self.dev)
            self.model.train_loop(train_dataset=train_dataset)

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
        budgets = budgets / self.max_benchmark_epochs
        real_budgets = np.array(real_budgets, dtype=np.single)
        configurations = np.array(configurations, dtype=np.single)

        configurations = torch.tensor(configurations, device=self.dev)
        budgets = torch.tensor(budgets, device=self.dev)
        hp_curves = torch.tensor(hp_curves, device=self.dev)

        train_data_fn = functools.partial(self.history_manager.prepare_dataset,
                                          curve_size_mode=self.meta.curve_size_mode)

        test_data = TabularDataset(
            X=configurations,
            budgets=budgets,
            curves=hp_curves,
        )

        mean_predictions, std_predictions = self.model.predict(test_data=test_data, train_data_fn=train_data_fn)

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
            if gv.IS_DYHPO:
                best_prediction_index = self.find_suggested_config_dyhpo(
                    mean_predictions,
                    std_predictions,
                    real_budgets,
                )
            else:
                best_prediction_index = self.find_suggested_config(
                    mean_predictions,
                    std_predictions,
                )
            """
            the best prediction index is not always matching with the actual hp index.
            Since when evaluating the acq function, we do not consider hyperparameter
            candidates that diverged or that are evaluated fully.
            """
            # actually do the mapping between the configuration indices and the best prediction index
            suggested_hp_index = hp_indices[best_prediction_index]

            # decide for what budget we will evaluate the most
            # promising hyperparameter configuration next.
            evaluated_budgets = self.history_manager.get_evaluated_budgets(suggested_hp_index)
            if len(evaluated_budgets) != 0:
                max_budget = max(evaluated_budgets)
                budget = max_budget + self.fantasize_step
                # this would only trigger if fantasize_step is bigger than 1
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

        if gv.IS_DYHPO:
            if self.minimization:
                hp_curve = np.subtract([self.max_value] * len(hp_curve), hp_curve)
                hp_curve = hp_curve.tolist()
        else:
            if not self.minimization:
                hp_curve = np.subtract([self.max_value] * len(hp_curve), hp_curve)
                hp_curve = hp_curve.tolist()

        if gv.IS_DYHPO:
            best_curve_value = max(hp_curve)
        else:
            best_curve_value = min(hp_curve)

        self.history_manager.add(hp_index, b, hp_curve)

        if (self.best_value_observed > best_curve_value and not gv.IS_DYHPO) or \
            (self.best_value_observed < best_curve_value and gv.IS_DYHPO):
            self.best_value_observed = best_curve_value
            self.no_improvement_patience = 0
            self.logger.info(f'New Incumbent value found '
                             f'{1 - best_curve_value if not self.minimization else best_curve_value}')
        else:
            self.no_improvement_patience += 1
            if self.no_improvement_patience >= self.no_improvement_threshold:
                self.train = True
                self.no_improvement_patience = 0
                self.logger.info(
                    'No improvement in the incumbent value threshold reached, '
                    'restarting training from scratch'
                )
        self.logger.debug(f"no_improvement_patience {self.no_improvement_patience}")
        if self.initial_random_index >= len(self.rand_init_conf_indices):

            if self.train:
                if self.pretrain:
                    # TODO Load the pregiven weights.
                    pass

                self._train_surrogate(pretrain=self.pretrain)
                if gv.IS_DYHPO:
                    if self.iterations_counter < self.initial_full_training_trials:
                        self.train = True
                    else:
                        self.train = False
                else:
                    if self.iterations_counter <= self.initial_full_training_trials:
                        self.train = True
                    else:
                        self.train = False
            else:
                self.refine_counter += 1
                self._train_surrogate(should_refine=True)

    def plot_pred_curve(self, hp_index, benchmark, method_budget, output_dir, prefix=""):
        if self.model is None:
            return

        real_curve = benchmark.get_curve(hp_index, self.max_benchmark_epochs)
        if gv.IS_DYHPO:
            if self.minimization:
                real_curve = np.subtract([self.max_value] * len(real_curve), real_curve)
                real_curve = real_curve.tolist()
        else:
            if not self.minimization:
                real_curve = np.subtract([self.max_value] * len(real_curve), real_curve)
                real_curve = real_curve.tolist()

        curves, max_budget = self.history_manager.get_curves(hp_index=hp_index,
                                                             curve_size_mode=self.meta.curve_size_mode)

        p_config = self.prepare_examples([hp_index])[0]
        p_config = torch.Tensor(p_config)
        p_config = p_config.expand(self.max_benchmark_epochs, -1)

        x_data = np.arange(1, self.max_benchmark_epochs + 1)
        p_budgets = torch.Tensor(x_data / self.max_benchmark_epochs)

        p_curve = torch.Tensor(curves)
        p_curve_last_row = p_curve[-1].unsqueeze(0)
        p_curve_num_repeats = self.max_benchmark_epochs - p_curve.size(0)
        repeated_last_row = p_curve_last_row.repeat_interleave(p_curve_num_repeats, dim=0)
        p_curve = torch.cat((p_curve, repeated_last_row), dim=0)

        plot_test_data = TabularDataset(
            X=p_config,
            budgets=p_budgets,
            curves=p_curve
        )

        train_data_fn = functools.partial(self.history_manager.prepare_dataset,
                                          curve_size_mode=self.meta.curve_size_mode)

        mean_data, std_data = self.model.predict(test_data=plot_test_data, train_data_fn=train_data_fn)

        plt.clf()
        p = sns.lineplot(x=x_data, y=mean_data)

        p.axes.fill_between(x_data, mean_data + std_data, mean_data - std_data, alpha=0.3)

        p.plot(x_data[:max_budget], real_curve[:max_budget], 'k-')
        p.plot(x_data[max_budget:], real_curve[max_budget:], 'k--')

        file_path = os.path.join(output_dir,
                                 f"{prefix}surrogatebudget_{method_budget}_budget_{max_budget}_hpindex_{hp_index}")
        plt.savefig(file_path, dpi=100)

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

        hp_indices, hp_budgets, real_budgets, hp_curves = self.history_manager.generate_candidate_configurations(
            predict_mode=self.meta.predict_mode,
            curve_size_mode=self.meta.curve_size_mode
        )
        configurations = self.prepare_examples(hp_indices)

        return configurations, hp_indices, hp_budgets, real_budgets, hp_curves

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

    def acq_dyhpo(
        self,
        best_value: float,
        mean: float,
        std: float,
        explore_factor: Optional[float] = 0.25,
        acq_fc: str = 'ei',
    ) -> float:
        """
        The acquisition function that will be called
        to evaluate the score of a hyperparameter configuration.

        Parameters
        ----------
        best_value: float
            Best observed function evaluation. Individual per fidelity.
        mean: float
            Point mean of the posterior process.
        std: float
            Point std of the posterior process.
        explore_factor: float
            The exploration factor for when ucb is used as the
            acquisition function.
        ei_calibration_factor: float
            The factor used to calibrate expected improvement.
        acq_fc: str
            The type of acquisition function to use.

        Returns
        -------
        acq_value: float
            The value of the acquisition function.
        """
        if acq_fc == 'ei':
            if std == 0:
                return 0
            z = (mean - best_value) / std
            acq_value = (mean - best_value) * norm.cdf(z) + std * norm.pdf(z)
        elif acq_fc == 'ucb':
            acq_value = mean + explore_factor * std
        elif acq_fc == 'thompson':
            acq_value = np.random.normal(mean, std)
        elif acq_fc == 'exploit':
            acq_value = mean
        else:
            raise NotImplementedError(
                f'Acquisition function {acq_fc} has not been'
                f'implemented',
            )

        return acq_value

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

    def find_suggested_config_dyhpo(
        self,
        mean_predictions: np.ndarray,
        mean_stds: np.ndarray,
        budgets: List,
    ) -> int:
        """
        Find the hyperparameter configuration that has the highest score
        with the acquisition function.

        Args:
            mean_predictions: The mean predictions of the posterior.
            mean_stds: The mean standard deviations of the posterior.
            budgets: The next budgets that the hyperparameter configurations
                will be evaluated for.

        Returns:
            best_index: The index of the hyperparameter configuration with the
                highest score.
        """
        highest_acq_value = np.NINF
        best_index = -1

        index = 0
        for mean_value, std in zip(mean_predictions, mean_stds):
            budget = int(budgets[index])
            best_value = self.history_manager.calculate_fidelity_ymax_dyhpo(budget)
            acq_value = self.acq_dyhpo(best_value, mean_value, std, acq_fc='ei')
            if acq_value > highest_acq_value:
                highest_acq_value = acq_value
                best_index = index

            index += 1

        return best_index
