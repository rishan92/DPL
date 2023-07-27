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
import math

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
    numpy_to_torch_apply, weighted_spearman
from scipy.stats import spearmanr
import properscoring as ps
from src.history.fidelity_manager import FidelityManager

warnings.filterwarnings('ignore')


class HyperparameterOptimizer(BaseHyperparameterOptimizer):
    model_types = {
        'power_law': EnsembleModel,
        'dyhpo': DyHPOModel,
    }

    def __init__(
        self,
        hp_candidates: np.ndarray,
        fidelity_manager: FidelityManager = None,
        surrogate_name: str = 'power_law',
        seed: int = 11,
        max_budgets: Union[int, Dict] = 52,
        min_budgets: Union[int, Dict] = 1,
        fantasize_step: Union[int, Dict] = 1,
        minimization: bool = True,
        total_budget: int = 1000,
        device: str = None,
        output_path: Path = '.',
        dataset_name: str = 'unknown',
        pretrain: bool = False,
        backbone: str = 'power_law',
        max_value: float = 100,
        min_value: float = 0,
        fill_value: str = 'zero',
        benchmark: BaseBenchmark = None,
        pred_trend_path: Path = None,
        fidelity_names=None,
        **kwargs
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

        super().__init__()

        self.model_type = surrogate_name
        self.model_class: Union[Type[EnsembleModel], Type[DyHPOModel]] = HyperparameterOptimizer.model_types[
            surrogate_name]
        self.model = None

        assert self.meta is not None, "Meta parameters are not set"

        self.total_budget = total_budget
        self.fill_value = fill_value
        self.max_value = max_value
        self.min_value = min_value
        self.backbone = backbone
        self.benchmark = benchmark
        self.pred_trend_path = pred_trend_path
        self.fidelity_manager = fidelity_manager

        self.pretrained_path = output_path / 'power_law' / f'checkpoint_{seed}.pth'

        if device is None:
            self.dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.dev = torch.device(device)

        self.hp_candidates: NDArray[np.float32] = hp_candidates.astype(dtype=np.float32)

        self.minimization = minimization
        self.seed = seed

        self.logger = logger

        # with what percentage configurations will be taken randomly instead of being sampled from the model
        self.fraction_random_configs = self.meta.fraction_random_configs
        # self.iteration_probabilities = np.random.rand(self.total_budget)
        if fidelity_names is None:
            fidelity_names = ['epochs']
        if not isinstance(fidelity_names, List):
            fidelity_names = [fidelity_names]
        self.fidelity_names: List = fidelity_names

        if isinstance(max_budgets, Dict):
            self.max_budgets: Dict = max_budgets
            self.min_budgets: Dict = min_budgets
            self.fantasize_step: Dict = fantasize_step
        else:
            self.max_budgets: Dict = {fidelity_names[0]: max_budgets}
            self.min_budgets: Dict = {fidelity_names[0]: min_budgets}
            self.fantasize_step: Dict = {fidelity_names[0]: fantasize_step}

        self.pretrain = pretrain

        initial_configurations_nr = 1
        conf_individual_budget = 1
        init_conf_indices = np.random.choice(self.hp_candidates.shape[0], initial_configurations_nr, replace=False)

        init_budgets = [{fidelity_name: self.min_budgets[fidelity_name] + i * self.fantasize_step[fidelity_name]
                         for fidelity_name in self.fidelity_names} for i in range(0, conf_individual_budget)]

        init_budgets = list(range(0, conf_individual_budget))

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

        # the number of initial points for which we will retrain fully from scratch
        # This is basically equal to the dimensionality of the search space + 1.
        self.initial_full_training_trials = self.meta.initial_full_training_trials

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

        max_steps = [((self.max_budgets[fidelity_name] - self.min_budgets[fidelity_name]) / self.fantasize_step[
            fidelity_name]) + 1 for fidelity_name in self.fidelity_names]
        max_steps = max(max_steps)
        self.no_improvement_threshold = int(max_steps + 0.2 * max_steps)
        self.no_improvement_patience = 0

        self.checkpoint_path = output_path / 'checkpoints' / f'{dataset_name}' / f'{self.seed}'
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        self.target_normalization_value = 1
        if self.meta.target_normalization_range is not None:
            self.target_normalization_range = self.meta.target_normalization_range
        else:
            self.target_normalization_range = [0, 1]

        self.target_normalization_inverse_fn = None
        self.target_normalization_std_inverse_fn = None

        self.history_manager = HistoryManager(
            hp_candidates=self.hp_candidates,
            fidelity_manager=self.fidelity_manager,
            max_budgets=self.max_budgets,
            min_budgets=self.min_budgets,
            fill_value=self.fill_value,
            use_learning_curve=self.model_class.meta_use_learning_curve,
            use_learning_curve_mask=self.model_class.meta_use_learning_curve_mask,
            fantasize_step=self.fantasize_step,
            use_target_normalization=self.meta.use_target_normalization,
            target_normalization_range=self.target_normalization_range,
            model_output_normalization=self.model_class.meta_output_act_func,
            use_scaled_budgets=self.meta.use_scaled_budgets,
            cnn_kernel_size=self.model_class.meta_cnn_kernel_size,
            use_sample_weights=self.model_class.meta_use_sample_weights,
            use_sample_weight_by_budget=self.model_class.meta_use_sample_weight_by_budget,
            sample_weight_by_budget_strategy=self.model_class.meta_sample_weight_by_budget_strategy,
            use_sample_weight_by_label=self.model_class.meta_use_sample_weight_by_label,
            use_y_constraint_weights=self.model_class.meta_use_y_constraint_weights,
            fidelity_names=self.fidelity_names,
        )

        self.real_curve_targets_map_pd: Optional[pd.DataFrame] = None
        self.prediction_params_pd: Optional[pd.DataFrame] = None

        self.surrogate_budget = 0

        # Inverse function is only used for plotting
        self.model_output_normalization_inverse_fn = None
        inverse_torch_class = get_inverse_function_class(self.model_class.meta_output_act_func)
        if inverse_torch_class:
            inverse_torch_fn = inverse_torch_class()
            self.model_output_normalization_inverse_fn = partial(numpy_to_torch_apply, torch_function=inverse_torch_fn)

        if gv.PLOT_ACQ or gv.PLOT_PRED_TREND:
            self.benchmark_labels = np.ones(shape=(self.hp_candidates.shape[0],))
            for i in range(self.hp_candidates.shape[0]):
                best_value = self.benchmark.get_curve_best(i)
                self.benchmark_labels[i] = best_value
            if not self.minimization:
                self.benchmark_labels = self.max_value - self.benchmark_labels
        if self.meta.check_model:
            self.check_training()

    @staticmethod
    def get_default_meta(model_class):
        if model_class == EnsembleModel:
            hp = {
                'fraction_random_configs': 0.1,
                'initial_full_training_trials': 10,
                'predict_mode': 'end_budget',
                'curve_size_mode': 'fixed',
                'acq_mode': 'ei',
                'acq_best_value_mode': 'normal',
                'use_target_normalization': False,
                'target_normalization_range': [0.1, 0.9],
                'use_scaled_budgets': True,
                'use_exploitation_sampling': False,
                'reset_refine_optimizer': True,
                'acq_explore_factor': 0,
                'acq_explore_strategy': 'constant',  # 'linear' 'exponential' 'cosine_annealing'
            }
        elif model_class == DyHPOModel:
            hp = {
                'fraction_random_configs': 0.1,
                'initial_full_training_trials': 10,
                'predict_mode': 'next_budget',  # 'end_budget',  #
                'curve_size_mode': 'variable',  # 'fixed',
                'acq_mode': 'ei',
                'acq_best_value_mode': 'mf',  # 'normal',  #    mf - multi-fidelity, normal, None
                'use_target_normalization': False,
                'target_normalization_range': [0.1, 0.9],
                'use_scaled_budgets': True,
                'use_exploitation_sampling': False,
                'reset_refine_optimizer': True,
                'acq_explore_factor': 0,
                'acq_explore_strategy': 'constant',
            }
        else:
            raise NotImplementedError(f"{model_class=}")

        hp["check_model"] = False
        hp["check_model_predict_mode"] = 'best'  # 'end'
        hp["check_model_train_mode"] = 'exp'
        hp["validation_configuration_ratio"] = 0.95
        hp['validation_curve_ratio'] = 0.98
        hp['validation_curve_prob'] = 0.95

        return hp

    @classmethod
    def set_meta(cls, config=None, **kwargs):
        surrogate_name = kwargs.pop('surrogate_name', None)
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

        train_dataset = self.history_manager.get_train_dataset(curve_size_mode=self.meta.curve_size_mode)

        if pretrain:
            should_refine = True,
            should_weight_last_sample = False

        last_sample = self.history_manager.get_last_sample(curve_size_mode=self.meta.curve_size_mode)

        if should_refine:
            return_state, _ = self.model.train_loop(
                train_dataset=train_dataset,
                should_refine=should_refine,
                reset_optimizer=self.meta.reset_refine_optimizer,
                last_sample=last_sample
            )
        else:
            if self.model is not None:
                self.model.reset()
            self.model = self.model_class(
                nr_features=train_dataset.X.shape[1],
                checkpoint_path=self.checkpoint_path,
                seed=self.seed,
                total_budget=self.total_budget,
                surrogate_budget=self.surrogate_budget,
                nr_fidelity=len(self.fidelity_names)
            )
            self.model.to(self.dev)
            return_state, _ = self.model.train_loop(train_dataset=train_dataset)

        return return_state

    def _predict(self) -> Tuple[
        NDArray[np.float32], NDArray[np.float32], NDArray[int], List[Dict], Optional[Dict], NDArray[np.float32]]:
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
        train_data = self.history_manager.get_train_dataset(curve_size_mode=self.meta.curve_size_mode)

        test_data, hp_indices, real_budgets = self.history_manager.get_candidate_configurations_dataset(
            predict_mode=self.meta.predict_mode,
            curve_size_mode=self.meta.curve_size_mode
        )

        mean_predictions, std_predictions, predict_infos, model_std_predictions = self.model.predict(
            test_data=test_data,
            train_data=train_data)

        # # order between model_output_normalization_inverse_fn and use_target_normalization should be reversed
        # # to match the order applied in making the training data
        # if self.model_output_normalization_inverse_fn:
        #     mean_predictions = self.model_output_normalization_inverse_fn(mean_predictions)
        #     std_predictions = self.model_output_normalization_inverse_fn(std_predictions)
        #     if predict_infos is not None and 'pl_output' in predict_infos:
        #         predict_infos['pl_output'] = self.model_output_normalization_inverse_fn(predict_infos['pl_output'])

        if self.meta.use_target_normalization:
            mean_predictions = self.target_normalization_inverse_fn(mean_predictions)
            std_predictions = self.target_normalization_std_inverse_fn(std_predictions)
            if model_std_predictions is not None:
                model_std_predictions = self.target_normalization_std_inverse_fn(model_std_predictions)
            if predict_infos is not None and 'pl_output' in predict_infos:
                predict_infos['pl_output'] = self.target_normalization_inverse_fn(predict_infos['pl_output'])

        return mean_predictions, std_predictions, hp_indices, real_budgets, predict_infos, model_std_predictions

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
                f'Not enough configurations to build a model at iteration. \n'
                'Returning randomly sampled configuration'
            )
            suggested_hp_index = self.rand_init_conf_indices[self.initial_random_index]
            # budget = self.rand_init_budgets[self.initial_random_index]
            self.initial_random_index += 1
        else:
            mean_predictions, std_predictions, hp_indices, real_budgets, predict_infos, mean_model_stds = self._predict()

            evaluated_mask = None
            if hasattr(self.meta, 'use_exploitation_sampling') and self.meta.use_exploitation_sampling:
                ex_sampling_probability = self.surrogate_budget / self.total_budget
                random_number = np.random.uniform(0, 1)
                if random_number < ex_sampling_probability:
                    evaluated_indices = self.history_manager.get_evaluted_indices()
                    if len(evaluated_indices) != 0:
                        evaluated_indices = np.array(evaluated_indices)
                        all_hp_indices = np.array(hp_indices)
                        evaluated_mask = np.isin(all_hp_indices, evaluated_indices)

            acq_explore_factor = 0
            if hasattr(self.meta, 'acq_explore_factor') and \
                self.meta.acq_explore_factor is not None and self.meta.acq_explore_factor != 0:
                acq_explore_factor = self.meta.acq_explore_factor
                if hasattr(self.meta, 'acq_explore_strategy') and \
                    self.meta.acq_explore_strategy is not None and self.meta.acq_explore_strategy != 'constant':
                    if self.meta.acq_explore_strategy == 'linear':
                        acq_explore_factor = self.meta.acq_explore_factor * (
                            1 - self.surrogate_budget / self.total_budget)
                    elif self.meta.acq_explore_strategy == 'exponential':
                        gamma = math.exp(math.log(1e-6 / self.meta.acq_explore_factor) / self.total_budget)
                        acq_explore_factor = math.pow(gamma, self.surrogate_budget)
                    elif self.meta.acq_explore_strategy == 'cosine_annealing':
                        acq_explore_factor = 0.5 * self.meta.acq_explore_factor * (
                            1 + math.cos(self.surrogate_budget * math.pi / self.total_budget))
                    else:
                        raise NotImplementedError

            best_prediction_index = self.find_suggested_config(
                mean_predictions,
                std_predictions,
                budgets=real_budgets,
                acq_mode=self.meta.acq_mode,
                acq_best_value_mode=self.meta.acq_best_value_mode,
                exploitation_mask=evaluated_mask,
                hp_indices=hp_indices,
                acq_explore_factor=acq_explore_factor,
                mean_model_stds=mean_model_stds
            )

            """
            the best prediction index is not always matching with the actual hp index.
            Since when evaluating the acq function, we do not consider hyperparameter
            candidates that diverged or that are evaluated fully.
            """
            # actually do the mapping between the configuration indices and the best prediction index
            suggested_hp_index: int = hp_indices[best_prediction_index]

            # decide for what budget we will evaluate the most
            # promising hyperparameter configuration next.
            evaluated_budgets = self.history_manager.get_evaluated_budgets(suggested_hp_index)
            if len(evaluated_budgets) != 0:
                # max_budget = max(evaluated_budgets)
                # budget = max_budget + self.fantasize_step[self.fidelity_name]
                # # this would only trigger if fantasize_step is bigger than 1
                # if budget > self.max_budgets[self.fidelity_name]:
                #     budget = self.max_budgets[self.fidelity_name]
                max_budget = evaluated_budgets[-1]
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
                budget = next_budget
            else:
                budget = {
                    fidelity_name: self.min_budgets[fidelity_name] for fidelity_name in self.fidelity_names
                }
            if gv.PLOT_PRED_CURVES and predict_infos is not None:
                if self.prediction_params_pd is None:
                    info_names = list(predict_infos.keys())
                    column_indexes = pd.MultiIndex.from_product([
                        hp_indices,
                        info_names
                    ], names=['hp_index', 'parameter_type'])
                    self.prediction_params_pd = pd.DataFrame(
                        index=np.arange(1, self.total_budget),
                        columns=column_indexes
                    )

                for k, v in predict_infos.items():
                    self.prediction_params_pd.loc[self.iterations_counter, (hp_indices, k)] = v

        fidelity_id = self.fidelity_manager.get_next_fidelity_id(configuration_id=suggested_hp_index)

        suggest_time_end = time.time()
        self.suggest_time_duration = suggest_time_end - suggest_time_start

        return suggested_hp_index, fidelity_id

    def observe(self, hp_index: int, budget: List[Tuple[int]], hp_curve: List):
        """Receive information regarding the performance of a hyperparameter
        configuration that was suggested.

        Args:
            hp_index: int
                The index of the evaluated hyperparameter configuration.
            budget: int
                The budget for which the hyperparameter configuration was evaluated.
            hp_curve: List
                The performance of the hyperparameter configuration.
        """
        self.surrogate_budget += 1

        for index, curve_element in enumerate(hp_curve):
            if np.isnan(curve_element):
                self.diverged_configs.add(hp_index)
                # only use the non-nan part of the curve and the corresponding
                # budget to still have the information in the network
                hp_curve = hp_curve[0:index + 1]
                b = index
                break

        if not self.minimization:
            hp_curve = self.max_value - np.array(hp_curve)
            hp_curve = hp_curve.tolist()

        best_curve_value = min(hp_curve)

        self.history_manager.add(hp_index, budget, hp_curve)

        if self.best_value_observed > best_curve_value:
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

        if gv.PLOT_ACQ and gv.IS_WANDB:
            wandb.log({
                "acq/training_from_scratch": 1.0 if self.train else 0.0
            })

        if self.initial_random_index >= len(self.rand_init_conf_indices):
            if self.train:
                self.target_normalization_value = self.history_manager.set_target_normalization_value()
                gap = self.target_normalization_range[1] - self.target_normalization_range[0]
                self.target_normalization_inverse_fn = \
                    lambda x: (x - self.target_normalization_range[0]) * self.target_normalization_value / gap
                self.target_normalization_std_inverse_fn = lambda x: x * self.target_normalization_value / gap

                if self.pretrain:
                    # TODO Load the pregiven weights.
                    pass

                return_state = self._train_surrogate(pretrain=self.pretrain)

                if self.iterations_counter <= self.initial_full_training_trials:
                    self.train = True
                else:
                    self.train = False
            else:
                self.refine_counter += 1
                return_state = self._train_surrogate(should_refine=True)

            # If the training has failed, restart training
            if return_state is not None and return_state < 0:
                self.train = True

    @staticmethod
    def acq(
        best_values: np.ndarray,
        mean_predictions: np.ndarray,
        std_predictions: np.ndarray,
        explore_factor: float = 0.25,
        acq_mode: str = 'ei',
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
        if acq_mode == 'ei':
            difference = best_values - mean_predictions - explore_factor

            zero_std_indicator = np.zeros_like(std_predictions, dtype=bool)
            zero_std_indicator[std_predictions == 0] = True
            not_zero_std_indicator = np.invert(zero_std_indicator)
            z = np.divide(difference, std_predictions, where=not_zero_std_indicator)
            z[zero_std_indicator] = 0

            acq_values = np.add(np.multiply(difference, norm.cdf(z)), np.multiply(std_predictions, norm.pdf(z)))
        elif acq_mode == 'ucb':
            # we are working with error rates so we multiply the mean with -1
            acq_values = np.add(-1 * mean_predictions, explore_factor * std_predictions)
        elif acq_mode == 'thompson':
            acq_values = np.random.normal(mean_predictions, std_predictions)
        elif acq_mode == 'exploit':
            acq_values = mean_predictions
        else:
            raise NotImplementedError(
                f'Acquisition function {acq_mode} has not been'
                f'implemented',
            )

        return acq_values

    def find_suggested_config(
        self,
        mean_predictions: NDArray[np.float32],
        mean_stds: NDArray[np.float32],
        budgets: List[Dict] = None,
        acq_mode: str = 'ei',
        acq_best_value_mode: str = None,
        exploitation_mask=None,
        hp_indices=None,
        acq_explore_factor=0,
        mean_model_stds: Optional[NDArray[np.float32]] = None,
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

        if acq_best_value_mode == 'mf':
            best_values_cache = {}
            best_values = np.empty(shape=len(budgets), dtype=np.float32)
            for i, budget in enumerate(budgets):
                if budget not in best_values_cache:
                    best_values_cache[budget] = self.history_manager.calculate_fidelity_ymax_dyhpo(budget)
                best_values[i] = best_values_cache[budget]
        else:
            best_values = np.full_like(mean_predictions, self.best_value_observed)

        if self.meta_uncertainty_combine_mode == 'before' and mean_model_stds is not None:
            data_stds = mean_stds - mean_model_stds
            mean_stds = self.meta_model_uncertainty_factor * mean_model_stds + \
                        self.meta_data_uncertainty_factor * data_stds
            mean_stds[mean_stds < 0] = 0

        if self.meta_uncertainty_combine_mode == 'after' and mean_model_stds is not None:
            data_stds = mean_stds - mean_model_stds
            data_stds[data_stds < 0] = 0
            model_acq_func_values = self.acq(
                best_values,
                mean_predictions,
                mean_model_stds,
                acq_mode=acq_mode,
                explore_factor=acq_explore_factor,
            )
            data_acq_func_values = self.acq(
                best_values,
                mean_predictions,
                data_stds,
                acq_mode=acq_mode,
                explore_factor=acq_explore_factor,
            )
            acq_func_values = self.meta_model_uncertainty_factor * model_acq_func_values + \
                              self.meta_data_uncertainty_factor * data_acq_func_values
        else:
            acq_func_values = self.acq(
                best_values,
                mean_predictions,
                mean_stds,
                acq_mode=acq_mode,
                explore_factor=acq_explore_factor,
            )

        if gv.PLOT_ACQ and gv.IS_WANDB:
            true_labels = self.benchmark_labels[hp_indices]
            mean_correlation, _ = spearmanr(mean_predictions, true_labels, nan_policy='raise')
            acq_correlation, _ = spearmanr(acq_func_values, -1 * true_labels, nan_policy='raise')

            w_mean_correlation = weighted_spearman(y_pred=mean_predictions, y_true=true_labels)
            w_acq_correlation = weighted_spearman(y_pred=acq_func_values, y_true=-1 * true_labels)

            min_label = np.min(true_labels)
            mean_regret = true_labels[np.argmin(mean_predictions)] - min_label
            acq_regret = true_labels[np.argmax(acq_func_values)] - min_label

            abs_residuals = np.abs(true_labels - mean_predictions)
            coverage_1std = np.mean(abs_residuals <= mean_stds) - 0.68
            coverage_2std = np.mean(abs_residuals <= 2 * mean_stds) - 0.95
            coverage_3std = np.mean(abs_residuals <= 3 * mean_stds) - 0.997

            crps_score = ps.crps_gaussian(true_labels, mu=mean_predictions, sig=mean_stds)
            crps_score = crps_score.mean()

            wandb.log({
                "acq/mean_correlation": mean_correlation,
                "acq/acq_correlation": acq_correlation,
                "acq/weighted_mean_correlation": w_mean_correlation,
                "acq/weighted_acq_correlation": w_acq_correlation,
                "acq/mean_regret": mean_regret,
                "acq/acq_regret": acq_regret,
                "acq/crps": crps_score,
                "acq/std_coverage_1sigma": coverage_1std,
                "acq/std_coverage_2sigma": coverage_2std,
                "acq/std_coverage_3sigma": coverage_3std,
                "acq/epoch": self.surrogate_budget,
            })

        if gv.PLOT_PRED_TREND:
            true_labels = self.benchmark_labels[hp_indices]
            plt.clf()

            difference = mean_predictions - true_labels
            g = sns.jointplot(x=difference, y=true_labels)
            g.plot_joint(sns.kdeplot, color="r", zorder=0, levels=6)
            g.plot_marginals(sns.rugplot, color="r", height=-.15, clip_on=False)

            plt.tight_layout()
            file_path = self.pred_trend_path / f"mean_distributions_surrogateBudget_{self.surrogate_budget}"
            plt.savefig(file_path, dpi=200)

            plt.close()

        if exploitation_mask is not None:
            acq_func_values[~exploitation_mask] = np.NINF

        max_value_index = np.argmax(acq_func_values)
        max_value_index = int(max_value_index)

        return max_value_index

    def plot_pred_curve(self, hp_index: int, benchmark: BaseBenchmark, surrogate_budget: int, output_dir: Path,
                        prefix: str = ""):
        if self.model is None:
            return

        # real_curves, real_curve_budgets = benchmark.get_curve(hp_index, self.max_budgets)
        #
        # if not self.minimization:
        #     real_curves = self.max_value - np.array(real_curves)

        train_data = self.history_manager.get_train_dataset(curve_size_mode=self.meta.curve_size_mode)

        prediction_data = []
        for fidelity_name in self.fidelity_names:
            pred_test_data, real_budgets, max_train_budget = self.history_manager.get_predict_curves_dataset(
                hp_index=hp_index,
                curve_size_mode=self.meta.curve_size_mode,
                fidelity_name=fidelity_name
            )

            mean_data, std_data, predict_infos, model_std_data = self.model.predict(test_data=pred_test_data,
                                                                                    train_data=train_data)

            # order between model_output_normalization_inverse_fn and use_target_normalization should be reversed
            # to match the order applied in making the training data. This is only calculated for plotting, since
            # standard deviation calculation have no simple solution.
            if self.model_type == "dyhpo" and self.model_output_normalization_inverse_fn:
                mean_data = self.model_output_normalization_inverse_fn(mean_data)
                # std_data = self.model_output_normalization_inverse_fn(std_data)
                if predict_infos is not None and 'pl_output' in predict_infos:
                    predict_infos['pl_output'] = self.model_output_normalization_inverse_fn(predict_infos['pl_output'])

            if self.meta.use_target_normalization:
                mean_data = self.target_normalization_inverse_fn(mean_data)
                std_data = self.target_normalization_std_inverse_fn(std_data)
                if predict_infos is not None and 'pl_output' in predict_infos:
                    predict_infos['pl_output'] = self.target_normalization_inverse_fn(predict_infos['pl_output'])

            # real_curve = [real_curves[real_curve_budgets.index(v)] for v in real_budgets]
            real_curve = []
            for real_budget in real_budgets:
                performance = benchmark.get_performance(hp_index, real_budget)
                real_curve.append(performance)

            if not self.minimization:
                real_curve = self.max_value - np.array(real_curve)

            prediction_data.append({
                'max_train_budget': max_train_budget,
                'real_budgets': real_budgets,
                'mean_data': mean_data,
                'std_data': std_data,
                'predict_infos': predict_infos,
                'model_std_data': model_std_data,
                'real_curve': real_curve,
            })

        # import torch.nn.functional as F
        # val_mae = F.l1_loss(torch.from_numpy(mean_data), torch.tensor(real_curve), reduction='mean')
        # print("val", val_mae)

        predict_infos_exist = prediction_data[-1]['predict_infos'] is not None and \
                              len(prediction_data[-1]['predict_infos']) > 1
        plt.clf()
        if predict_infos_exist:
            nrows, ncols, figsize = 1, 2, (10, 6)
        else:
            nrows, ncols, figsize = len(self.fidelity_names), 1, (6.4, 4.8 * len(self.fidelity_names))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        max_train_budget_str = ' '.join(f'{k} {v}' for k, v in prediction_data[-1]['max_train_budget'].items())
        fig.suptitle(
            f'hp index {hp_index} at surrogate budget {surrogate_budget} and budget {max_train_budget_str}'
        )

        if predict_infos_exist and len(self.fidelity_names) == 1:
            predict_curve_axes = axes[0]
            param_axes = axes[1]
        elif predict_infos_exist:
            predict_curve_axes = axes[0]
            param_axes = axes[1]
        else:
            predict_curve_axes = axes
            param_axes = None

        for i, fidelity_name in enumerate(self.fidelity_names):
            real_budgets = prediction_data[i]['real_budgets']
            mean_data = prediction_data[i]['mean_data']
            std_data = prediction_data[i]['std_data']
            predict_infos = prediction_data[i]['predict_infos']
            real_curve = prediction_data[i]['real_curve']
            # max_train_budget_limit = real_budgets.index(prediction_data[i]['max_train_budget'])
            max_train_budget_limit = 0

            real_budgets = self.fidelity_manager.get_fidelities(
                fidelity_ids=real_budgets, is_normalized=False, return_dict=True,
            )
            x_data = [v[fidelity_name] for v in real_budgets]
            sns.lineplot(x=x_data, y=mean_data, ax=predict_curve_axes[i], color='blue', label='mean prediction')
            if predict_infos_exist:
                sns.lineplot(x=x_data, y=predict_infos['pl_output'], ax=predict_curve_axes[i], color='red',
                             label='power law output')
            predict_curve_axes[i].set_xlabel(f'Budget {fidelity_name}')

            predict_curve_axes[i].fill_between(x_data, mean_data + std_data, mean_data - std_data, alpha=0.3)

            predict_curve_axes[i].plot(x_data[:max_train_budget_limit], real_curve[:max_train_budget_limit], 'k-')
            predict_curve_axes[i].plot(x_data[max_train_budget_limit:], real_curve[max_train_budget_limit:], 'k--')

        if predict_infos_exist:
            data = self.prediction_params_pd.loc[:, hp_index]
            sns.lineplot(data=data, ax=param_axes)
            param_axes.set_xlabel('Surrogate Budget')

        plt.tight_layout()
        file_path = \
            output_dir / f"{prefix}surrogateBudget_{surrogate_budget}_trainBudget_{max_train_budget_str}_hpIndex_{hp_index}.png"
        plt.savefig(file_path, dpi=200)

        plt.close()

    def plot_pred_dist(self, benchmark: BaseBenchmark, surrogate_budget: int, output_dir: Path, prefix: str = ""):
        if self.model is None:
            return

        test_data, hp_indices, real_budgets = self.history_manager.get_candidate_configurations_dataset(
            predict_mode=self.meta.predict_mode,
            curve_size_mode=self.meta.curve_size_mode
        )

        if self.real_curve_targets_map_pd is None:
            real_curve_targets = np.empty(shape=(len(hp_indices),), dtype=np.float32)
            for i, hp_index in enumerate(hp_indices):
                real_curve = benchmark.get_curve(hp_index, self.max_benchmark_epochs)

                if not self.minimization:
                    real_curve = self.max_value - np.array(real_curve)

                real_curve_targets[i] = min(real_curve)
            self.real_curve_targets_map_pd = pd.DataFrame(real_curve_targets, index=hp_indices)
        else:
            real_curve_targets = self.real_curve_targets_map_pd.iloc[hp_indices, 0]

        train_data = self.history_manager.get_train_dataset(curve_size_mode=self.meta.curve_size_mode)

        mean_data, std_data, predict_infos = self.model.predict(test_data=test_data, train_data=train_data)

        # order between model_output_normalization_inverse_fn and use_target_normalization should be reversed
        # to match the order applied in making the training data. This is only calculated for plotting, since
        # standard deviation calculation have no simple solution.
        if self.model_type == "dyhpo" and self.model_output_normalization_inverse_fn:
            mean_data = self.model_output_normalization_inverse_fn(mean_data)
            # std_data = self.model_output_normalization_inverse_fn(std_data)
            if predict_infos is not None and 'pl_output' in predict_infos:
                predict_infos['pl_output'] = self.model_output_normalization_inverse_fn(predict_infos['pl_output'])

        if self.meta.use_target_normalization:
            mean_data = self.target_normalization_inverse_fn(mean_data)
            # std_data = self.target_normalization_std_inverse_fn(std_data)
            if predict_infos is not None and 'pl_output' in predict_infos:
                predict_infos['pl_output'] = self.target_normalization_inverse_fn(predict_infos['pl_output'])

        difference = real_curve_targets - mean_data

        plt.clf()
        if predict_infos is not None:
            nrows, ncols, figsize = 3, 2, (10, 6)
        else:
            nrows, ncols, figsize = 1, 2, (10, 4)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        fig.suptitle(
            f'Distributions at surrogate budget {surrogate_budget}'
        )
        target_axes = axes[0, 0] if predict_infos is not None else axes[0]
        difference_axes = axes[0, 1] if predict_infos is not None else axes[1]

        bin_width = 0.001
        sns.histplot(data=real_curve_targets, kde=False, ax=target_axes, color='blue', label='Real Target', alpha=0.5,
                     binwidth=bin_width)
        sns.histplot(data=mean_data, kde=False, ax=target_axes, color='red', label='Predicted Target', alpha=0.5,
                     binwidth=bin_width)
        target_axes.set_title('Real & Predicted Targets')
        target_axes.legend(loc='upper right')

        sns.histplot(data=difference, kde=False, ax=difference_axes, label='Real Target - Predicted Target')
        difference_axes.set_title('Real Target - Predicted Target')

        if predict_infos is not None:
            sns.histplot(data=predict_infos['alpha'], kde=False, ax=axes[1, 0], label='alpha')
            axes[1, 0].set_title('alpha')
            sns.histplot(data=predict_infos['beta'], kde=False, ax=axes[1, 1], label='beta')
            axes[1, 1].set_title('beta')
            sns.histplot(data=predict_infos['gamma'], kde=False, ax=axes[2, 0], label='gamma')
            axes[2, 0].set_title('gamma')

        plt.tight_layout()
        file_path = output_dir / f"{prefix}distributions_surrogateBudget_{surrogate_budget}"
        plt.savefig(file_path, dpi=200)

        plt.close()

    def check_training(self):
        train_dataset, val_dataset, target_normalization_value = \
            self.history_manager.get_check_train_validation_dataset(
                curve_size_mode=self.meta.curve_size_mode,
                benchmark=self.benchmark,
                validation_configuration_ratio=self.meta.validation_configuration_ratio,
                validation_curve_ratio=self.meta.validation_curve_ratio,
                validation_mode=self.meta.check_model_predict_mode,
                check_model_train_mode=self.meta.check_model_train_mode,
                validation_curve_prob=self.meta.validation_curve_prob,
                seed=self.seed
            )
        self.target_normalization_value = target_normalization_value

        if self.model is not None:
            self.model.reset()
        self.model = self.model_class(
            nr_features=train_dataset.X.shape[1],
            checkpoint_path=self.checkpoint_path,
            seed=self.seed,
            total_budget=self.total_budget,
            surrogate_budget=1
        )
        if self.meta.use_target_normalization:
            gap = self.target_normalization_range[1] - self.target_normalization_range[0]
            self.target_normalization_inverse_fn = \
                lambda x: (x - self.target_normalization_range[0]) * self.target_normalization_value / gap
            self.target_normalization_std_inverse_fn = lambda x: x * self.target_normalization_value / gap

            self.model.set_target_normalization_inverse_function(
                fn=self.target_normalization_inverse_fn,
                std_fn=self.target_normalization_std_inverse_fn
            )
        self.model.to(self.dev)
        return_state, _ = self.model.train_loop(train_dataset=train_dataset, val_dataset=val_dataset)
        if return_state is not None and return_state < 0:
            print("Training failed. Restarting.")
            self.logger.warning("Training failed. Restarting.")

        sys.exit(0)

    @property
    def meta_uncertainty_combine_mode(self):
        if hasattr(self.model_class.meta, 'uncertainty_combine_mode'):
            return self.model_class.meta.uncertainty_combine_mode
        else:
            return None

    @property
    def meta_model_uncertainty_factor(self):
        if hasattr(self.model_class.meta, 'model_uncertainty_factor'):
            return self.model_class.meta.model_uncertainty_factor
        else:
            return 1

    @property
    def meta_data_uncertainty_factor(self):
        if hasattr(self.model_class.meta, 'data_uncertainty_factor'):
            return self.model_class.meta.data_uncertainty_factor
        else:
            return 1
