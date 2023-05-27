import time
from copy import deepcopy
import os
from typing import Dict, Tuple, Any, Optional
from numpy.typing import NDArray
import numpy as np
import torch
from loguru import logger
import gpytorch
import wandb
from gpytorch.constraints import Interval, GreaterThan, LessThan
from types import SimpleNamespace
from pathlib import Path
import inspect
from torch.nn.utils import clip_grad_norm_
import itertools
from functools import partial
import torch.nn.functional as F
from torch.utils.data import Subset
import traceback
import math
from scipy.stats import spearmanr
import properscoring as ps

from src.models.deep_kernel_learning.base_feature_extractor import BaseFeatureExtractor
from src.models.deep_kernel_learning.gp_regression_model import GPRegressionModel
from src.models.base.base_pytorch_module import BasePytorchModule
from src.dataset.tabular_dataset import TabularDataset
from src.utils.utils import get_class_from_package, get_class_from_packages
import src.models.deep_kernel_learning
from src.models.base.meta import Meta
from src.utils.utils import get_class, classproperty
import global_variables as gv
from src.utils.torch_lr_finder import LRFinder
from src.data_loader.surrogate_data_loader import SurrogateDataLoader
from src.utils.utils import acq


class DyHPOModel(BasePytorchModule):
    """
    The DyHPO DeepGP model.
    """
    _global_epoch = 0
    _training_errors = 0
    _suggested_lr = None
    _nan_gradients = 0

    def __init__(
        self,
        nr_features,
        total_budget,
        surrogate_budget,
        checkpoint_path: Path = '.',
        seed=None
    ):
        """
        The constructor for the DyHPO model.

        Args:
            configuration: The configuration to be used
                for the different parts of the surrogate.
            device: The device where the experiments will be run on.
            output_path: The path where the intermediate/final results
                will be stored.
            seed: The seed that will be used to store the checkpoint
                properly.
        """
        super().__init__(nr_features=nr_features, seed=seed)
        self.feature_extractor_class = get_class_from_package(src.models.deep_kernel_learning,
                                                              self.meta.feature_class_name)
        self.gp_class = get_class_from_package(src.models.deep_kernel_learning,
                                               self.meta.gp_class_name)
        self.likelihood_class = get_class_from_packages([gpytorch.likelihoods, src.models.deep_kernel_learning],
                                                        self.meta.likelihood_class_name)
        self.mll_criterion_class = get_class_from_packages([gpytorch.mlls, src.models.deep_kernel_learning],
                                                           self.meta.mll_loss_function)
        self.power_law_criterion_class = get_class_from_packages([torch.nn, src.models.deep_kernel_learning],
                                                                 self.meta.power_law_loss_function)

        self.feature_extractor: BaseFeatureExtractor = self.feature_extractor_class(nr_features=nr_features, seed=seed)
        self.feature_output_size = self.get_feature_extractor_output_size(nr_features=nr_features)

        self.early_stopping_patience = self.meta.nr_patience_epochs
        self.seed = seed
        self.total_budget = total_budget
        self.surrogate_budget = surrogate_budget

        if self.meta.noise_lower_bound is not None and self.meta.noise_upper_bound is not None:
            self.noise_constraint = Interval(lower_bound=self.meta.noise_lower_bound,
                                             upper_bound=self.meta.noise_upper_bound)
        elif self.meta.noise_lower_bound is not None:
            self.noise_constraint = GreaterThan(lower_bound=self.meta.noise_lower_bound)
        elif self.meta.noise_upper_bound is not None:
            self.noise_constraint = LessThan(upper_bound=self.meta.noise_upper_bound)
        else:
            self.noise_constraint = None

        gp_input_size = self.feature_output_size

        self.likelihood: gpytorch.likelihoods.Likelihood = self.likelihood_class(noise_constraint=self.noise_constraint)
        self.model: gpytorch.models.GP = self.gp_class(
            input_size=gp_input_size,
            likelihood=self.likelihood,
            use_seperate_lengthscales=self.meta.use_seperate_lengthscales,
            use_scale_to_bounds=self.meta.use_scale_to_bounds
        )
        self.mll_criterion: gpytorch.mlls.MarginalLogLikelihood = self.mll_criterion_class(self.likelihood, self.model)
        self.power_law_criterion = self.power_law_criterion_class()

        self.optimizer = None
        self.lr_scheduler = None

        self.logger = logger

        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        self.checkpoint_file = self.checkpoint_path / 'checkpoint.pth'

        self.clip_gradients_value = None
        if hasattr(self.meta, 'clip_gradients') and self.meta.clip_gradients != 0:
            self.clip_gradients_value = self.meta.clip_gradients

        self.target_normalization_inverse_fn = None
        self.target_normalization_std_inverse_fn = None

        self.regularization_factor = 0
        if hasattr(self.meta, 'weight_regularization_factor') and self.meta.weight_regularization_factor != 0:
            self.regularization_factor = self.meta.weight_regularization_factor

        self.alpha_beta_constraint_factor = 0
        if hasattr(self.meta, 'alpha_beta_constraint_factor') and self.meta.alpha_beta_constraint_factor != 0:
            self.alpha_beta_constraint_factor = self.meta.alpha_beta_constraint_factor

        self.gamma_constraint_factor = 0
        if hasattr(self.meta, 'gamma_constraint_factor') and self.meta.gamma_constraint_factor != 0:
            self.gamma_constraint_factor = self.meta.gamma_constraint_factor

        self.output_constraint_factor = 0
        if hasattr(self.meta, 'output_constraint_factor') and self.meta.output_constraint_factor != 0:
            self.output_constraint_factor = self.meta.output_constraint_factor

        self.target_space_constraint_factor = 0
        if hasattr(self.meta, 'target_space_constraint_factor') and self.meta.target_space_constraint_factor != 0:
            self.target_space_constraint_factor = self.meta.target_space_constraint_factor

        self.l1_loss = torch.nn.L1Loss()

        self.l1_loss_factor = 0
        if hasattr(self.meta, 'l1_loss_factor') and self.meta.l1_loss_factor != 0:
            self.l1_loss_factor = self.meta.l1_loss_factor

        self.power_law_l1_loss_factor = 0
        if hasattr(self.meta, 'power_law_l1_loss_factor') and self.meta.power_law_l1_loss_factor != 0:
            self.power_law_l1_loss_factor = self.meta.power_law_l1_loss_factor

        self.mll_loss_factor = 1
        if hasattr(self.meta, 'mll_loss_factor'):
            self.mll_loss_factor = self.meta.mll_loss_factor

        if gv.IS_WANDB and gv.PLOT_GRADIENTS:
            wandb.watch([self.feature_extractor, self.model, self.likelihood], log='all', idx=0, log_freq=10)

    @staticmethod
    def get_default_meta(**kwargs) -> Dict[str, Any]:
        hp = {
            'batch_size': 64,
            'nr_patience_epochs': 10,
            'nr_epochs': 1000,
            'refine_nr_epochs': 50,
            'feature_class_name': 'FeatureExtractorTargetSpaceDYHPO',
            # 'FeatureExtractor',  #  'FeatureExtractorDYHPO',  # 'FeatureExtractorTargetSpaceDYHPO'
            'gp_class_name': 'GPRegressionPowerLawMeanModel',
            # 'GPRegressionPowerLawMeanModel',  #  'GPRegressionModel'
            'likelihood_class_name': 'GaussianLikelihood',
            'mll_loss_function': 'ExactMarginalLogLikelihood',
            'learning_rate': 1e-3,
            'refine_learning_rate': 1e-3,
            'power_law_loss_function': 'MSELoss',
            'power_law_loss_factor': 0.5,
            'l1_loss_factor': 0,
            'power_law_l1_loss_factor': 1,
            'mll_loss_factor': 0,
            'weight_regularization_factor': 0,
            'alpha_beta_constraint_factor': 0,
            'gamma_constraint_factor': 0,
            'output_constraint_factor': 0,
            'target_space_constraint_factor': 0,
            'noise_lower_bound': 1e-4,  # 1e-4,  #
            'noise_upper_bound': 1e-3,  # 1e-3,  #
            'use_seperate_lengthscales': False,
            'optimize_likelihood': False,
            'use_scale_to_bounds': False,
            'use_suggested_learning_rate': False,
            'use_weight_by_budget': False,
            'optimizer': 'Adam',
            'reset_on_divergence': False,
            'learning_rate_scheduler': None,
            # 'CosineAnnealingLR' 'LambdaLR' 'OneCycleLR' 'ExponentialLR'
            'learning_rate_scheduler_args': {
                'total_iters_factor': 1.0,
                'eta_min': 1e-6,
                'max_lr': 1e-4,
                'refine_max_lr': 1e-3,
                'exp_min': 1e-6,
                'refine_exp_min': 1e-6,
            },
        }

        return hp

    @classmethod
    def set_meta(cls, config=None, **kwargs):
        config = {} if config is None else config
        default_meta = cls.get_default_meta()
        meta = {**default_meta, **config}
        feature_model_class = get_class("src/models/deep_kernel_learning", meta['feature_class_name'])
        feature_model_config = feature_model_class.set_meta(config.get("feature_model", None))
        meta['feature_model'] = feature_model_config
        cls.meta = SimpleNamespace(**meta)
        return meta

    def set_optimizer(self, use_scheduler=False, **kwargs):
        is_refine = self.optimizer is not None
        if is_refine and hasattr(self.meta, 'refine_learning_rate') and self.meta.refine_learning_rate:
            learning_rate = self.meta.refine_learning_rate
        else:
            learning_rate = self.meta.learning_rate
        optimizer_class = get_class_from_package(torch.optim, self.meta.optimizer)
        if not self.meta.optimize_likelihood:
            self.optimizer = optimizer_class([
                {'params': self.model.parameters(), 'lr': learning_rate},
                {'params': self.feature_extractor.parameters(), 'lr': learning_rate}],
            )
        else:
            self.optimizer = optimizer_class(self.parameters(), lr=learning_rate)

        if use_scheduler:
            self.set_lr_scheduler(is_refine=is_refine, **kwargs)

    def set_lr_scheduler(self, is_refine: bool, **kwargs):
        if hasattr(self.meta, 'learning_rate_scheduler') and self.meta.learning_rate_scheduler:
            lr_scheduler_class = get_class_from_package(torch.optim.lr_scheduler, self.meta.learning_rate_scheduler)
            args = {}
            if hasattr(self.meta, 'learning_rate_scheduler_args') and self.meta.learning_rate_scheduler_args:
                args = self.meta.learning_rate_scheduler_args

            epochs = kwargs['epochs']
            if "total_iters_factor" in args:
                args["total_iters"] = epochs * args["total_iters_factor"]
                args["T_max"] = epochs * args["total_iters_factor"]
            else:
                args["total_iters"] = epochs
                args["T_max"] = epochs

            args["verbose"] = False
            args["epochs"] = epochs
            args["total_steps"] = epochs
            args["lr_lambda"] = lambda step: 1 - step / args["total_iters"]
            args["gamma"] = math.exp(math.log(args["exp_min"] / self.meta.learning_rate) / args["total_iters"])
            if is_refine and 'refine_max_lr' in args and args["refine_max_lr"]:
                args["max_lr"] = args["refine_max_lr"]
            if is_refine and 'refine_exp_min' in args and args["refine_exp_min"]:
                args["gamma"] = math.exp(
                    math.log(args["refine_exp_min"] / self.meta.refine_learning_rate) / args["total_iters"])

            arg_names = inspect.getfullargspec(lr_scheduler_class.__init__).args
            relevant_args = {k: v for k, v in args.items() if k in arg_names}
            self.lr_scheduler = lr_scheduler_class(self.optimizer, **relevant_args)

    def lr_finder_train(self, nr_epochs, train_dataloader=None, **kwargs):
        prev_epochs = self.meta.nr_epochs
        self.meta.nr_epochs = nr_epochs

        batch = next(train_dataloader)
        batch_examples, batch_labels, batch_budgets, batch_curves, batch_weights = batch

        train_dataset = TabularDataset(
            X=batch_examples,
            Y=batch_labels,
            budgets=batch_budgets,
            curves=batch_curves,
        )
        return_state, loss = self.train_loop(train_dataset=train_dataset, **kwargs)

        self.meta.nr_epochs = prev_epochs

        return return_state, loss

    def train_loop(self, train_dataset: TabularDataset, should_refine: bool = False, load_checkpoint: bool = False,
                   reset_optimizer=False, val_dataset: TabularDataset = None, is_lr_finder=False, **kwargs):
        """
        Train the surrogate model.

        Args:
            train_dataset: A Dataset which has the training examples, training features,
                training budgets and in the end the training curves.
            load_checkpoint: A flag whether to load the state from a previous checkpoint,
                or whether to start from scratch.
        """
        model_device = next(self.parameters()).device
        train_dataset.to(model_device)
        if val_dataset:
            val_dataset.to(model_device)

        if (gv.PLOT_SUGGEST_LR or self.meta.use_suggested_learning_rate) and not is_lr_finder:
            train_dataloader = SurrogateDataLoader(
                dataset=train_dataset, batch_size=64, shuffle=True, seed=self.seed, dev=model_device
            )
            DyHPOModel._suggested_lr = self.suggest_learning_rate(train_dataloader=train_dataloader)

        self.set_seed(self.seed)
        self.train()

        if load_checkpoint:
            try:
                self.load_checkpoint()
            except FileNotFoundError:
                self.logger.error(f'No checkpoint file found at: {self.checkpoint_file}'
                                  f'Training the GP from the beginning')

        if should_refine:
            nr_epochs = self.meta.refine_nr_epochs
            batch_size = 64
        else:
            nr_epochs = self.meta.nr_epochs
            batch_size = 64

        if reset_optimizer or self.optimizer is None:
            self.set_optimizer(epochs=nr_epochs, use_scheduler=True)

        if self.meta.use_suggested_learning_rate and DyHPOModel._suggested_lr is not None and not is_lr_finder:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = DyHPOModel._suggested_lr

        x_train = train_dataset.X
        train_budgets = train_dataset.budgets
        train_curves = train_dataset.curves
        y_train = train_dataset.Y

        initial_state = self.get_state()
        training_errored = False

        # where the mean squared error will be stored
        # when predicting on the train set
        mae_value = 0.0
        val_mae_value = 0.0
        is_nan_gradient = False
        loss_value = 0.0

        nr_examples_batch = x_train.size(dim=0)
        # if only one example in the batch, skip the batch.
        # Otherwise, the code will fail because of batchnorm
        if self.has_batchnorm_layers and nr_examples_batch == 1:
            return None, None

        for epoch_nr in range(0, nr_epochs):
            check_seed_torch = torch.random.get_rng_state().sum()
            if not is_lr_finder:
                DyHPOModel._global_epoch += 1
            is_nan_gradient = False

            # Zero backprop gradients
            self.optimizer.zero_grad()

            projected_x, predict_info = self.feature_extractor(x_train, train_budgets, train_curves)
            self.model.set_train_data(projected_x, y_train, strict=False)
            output: gpytorch.ExactMarginalLogLikelihood = self.model(projected_x)

            l1_norm = torch.tensor(0.0, requires_grad=True)
            if self.regularization_factor != 0:
                num_params = 0
                for param in self.parameters():
                    l1_norm = l1_norm + torch.norm(param, 1)
                    num_params += param.numel()
                l1_norm = l1_norm / num_params
                l1_norm = torch.max(torch.tensor(1), l1_norm) - 1

            if self.alpha_beta_constraint_factor != 0:
                alpha_plus_beta = predict_info['alpha'] + predict_info['beta']
                lower_loss = torch.clamp(-1 * alpha_plus_beta, min=0)
                upper_loss = torch.clamp(alpha_plus_beta - 1, min=0)
                alpha_beta_constraint_loss = torch.mean(lower_loss + upper_loss)
            else:
                alpha_beta_constraint_loss = torch.tensor(0.0, requires_grad=True)

            if self.gamma_constraint_factor != 0:
                gamma = predict_info['gamma']
                gamma_upper_bound = \
                    torch.log((1 - predict_info['alpha'] - torch.tensor(1e-4)) / (
                        predict_info['beta'] + torch.tensor(1e-4))) / torch.log(torch.tensor(51))
                lower_loss = torch.clamp(-1 * gamma, min=0)
                upper_loss = torch.clamp(gamma - gamma_upper_bound, min=0)
                gamma_constraint_loss = torch.mean(lower_loss + upper_loss)
            else:
                gamma_constraint_loss = torch.tensor(0.0, requires_grad=True)

            if self.target_space_constraint_factor != 0:
                y1: torch.Tensor = predict_info['y1']
                y2: torch.Tensor = predict_info['y2']
                alpha: torch.Tensor = predict_info['alpha']
                # mask = (y1 >= y2) & ((alpha <= y2) | (alpha >= y1)) | (y1 < y2) & ((alpha <= y1) | (alpha >= y2))
                mask = (y1 >= y2) & (alpha <= y2) | (y1 <= y2) & (alpha >= y2)
                target_space_constraint_loss = \
                    torch.where(mask, torch.tensor(0.0), torch.abs(alpha - ((y1 + y2) / 2)))
                target_space_constraint_loss = target_space_constraint_loss.mean()
            else:
                target_space_constraint_loss = torch.tensor(0.0, requires_grad=True)

            try:
                # Calc loss and backprop derivatives
                mll_loss = -self.mll_criterion(output, self.model.train_targets)

                prediction: gpytorch.distributions.Distribution = self.likelihood(output)
                power_law_output = projected_x[:, -1]
                mean_prediction = prediction.mean
                power_law_loss = self.power_law_criterion(mean_prediction, power_law_output)

                if self.power_law_l1_loss_factor != 0:
                    power_law_l1_loss = self.l1_loss(power_law_output, self.model.train_targets)
                else:
                    power_law_l1_loss = torch.tensor(0.0, requires_grad=True)

                if self.l1_loss_factor != 0:
                    l1_loss = self.l1_loss(mean_prediction, self.model.train_targets)
                else:
                    l1_loss = torch.tensor(0.0, requires_grad=True)

                if self.output_constraint_factor != 0:
                    lower_loss = torch.clamp(-1 * power_law_output, min=0)
                    upper_loss = torch.clamp(power_law_output - 1, min=0)
                    output_constraint_loss = torch.mean(lower_loss + upper_loss)
                else:
                    output_constraint_loss = torch.tensor(0.0, requires_grad=True)
                # diff_m = torch.sum(torch.abs(y_train - self.model.train_targets))
                # stds = prediction.stddev
                # diff = torch.abs(mean_prediction - y_train)
                # print(epoch_nr, stds[-1], diff[-1], stds, diff)
                loss = self.mll_loss_factor * mll_loss + \
                       self.meta.power_law_loss_factor * power_law_loss + \
                       self.regularization_factor * l1_norm + \
                       self.alpha_beta_constraint_factor * alpha_beta_constraint_loss + \
                       self.l1_loss_factor * l1_loss + \
                       self.power_law_l1_loss_factor * power_law_l1_loss + \
                       self.gamma_constraint_factor * gamma_constraint_loss + \
                       self.output_constraint_factor * output_constraint_loss + \
                       self.target_space_constraint_factor * target_space_constraint_loss

                mae = gpytorch.metrics.mean_absolute_error(prediction, self.model.train_targets)
                mse = gpytorch.metrics.mean_squared_error(prediction, self.model.train_targets)
                power_law_mae = gpytorch.metrics.mean_absolute_error(prediction, power_law_output.detach())

                mll_loss_value = mll_loss.detach().to('cpu').item()
                power_law_loss_value = power_law_loss.detach().to('cpu').item()
                loss_value = loss.detach().to('cpu').item()
                mae_value = mae.detach().to('cpu').item()
                mse_value = mse.detach().to('cpu').item()
                power_law_mae_value = power_law_mae.detach().to('cpu').item()
                lengthscale_value = self.model.covar_module.base_kernel.lengthscale[0, 0].detach().to('cpu').item()
                noise_value = self.model.likelihood.noise.detach().to('cpu').item()
                l1_loss_value = l1_loss.detach().to('cpu').item()

                loss.backward()

                if not is_nan_gradient:
                    for name, param in self.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any():
                                is_nan_gradient = True
                                break

                if self.clip_gradients_value:
                    if not self.meta.optimize_likelihood:
                        parameters = itertools.chain(self.model.parameters(), self.feature_extractor.parameters())
                    else:
                        parameters = self.parameters()
                    clip_grad_norm_(parameters, self.clip_gradients_value)

                self.optimizer.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step()

                if val_dataset:
                    means, stds, predict_infos, val_power_law_mae = self._batch_predict(
                        test_data=val_dataset,
                        train_data=train_dataset,
                        target_normalization_inverse_fn=self.target_normalization_inverse_fn,
                        target_normalization_std_inverse_fn=self.target_normalization_std_inverse_fn
                    )
                    means_tensor = torch.from_numpy(means).to(model_device)
                    labels = val_dataset.Y
                    val_mae = F.l1_loss(means_tensor, labels, reduction='mean')
                    val_mae_value = val_mae.detach().to('cpu').item()

                    labels = labels.detach().numpy()

                    abs_residuals = np.abs(means - labels)
                    val_correlation_value = np.corrcoef(stds, abs_residuals)
                    val_correlation_value = val_correlation_value[0, 1]

                    coverage_1std = np.mean(abs_residuals <= stds) - 0.68
                    coverage_2std = np.mean(abs_residuals <= 2 * stds) - 0.95
                    coverage_3std = np.mean(abs_residuals <= 3 * stds) - 0.997

                    crps_score = ps.crps_gaussian(labels, mu=means, sig=stds)
                    crps_score = crps_score.mean()

                    best_values = np.ones_like(means)
                    best_values[:] = np.mean(labels)
                    acq_func_values = acq(
                        best_values,
                        means,
                        stds,
                        acq_mode='ei',
                    )
                    mean_correlation, _ = spearmanr(means, labels, nan_policy='raise')
                    acq_correlation, _ = spearmanr(acq_func_values, labels, nan_policy='raise')

                    self.train()

                if is_nan_gradient and not is_lr_finder:
                    DyHPOModel._nan_gradients += 1

                if is_lr_finder:
                    continue

                self.logger.debug(
                    f'Epoch {epoch_nr} - MSE {mse_value}, '
                    f'Loss: {loss_value}, '
                    f'lengthscale: {lengthscale_value}, '
                    f'noise: {noise_value}, '
                )

                current_lr = self.optimizer.param_groups[0]['lr']

                wandb_data = {
                    "surrogate/dyhpo/training_loss": loss_value,
                    "surrogate/dyhpo/training_mll_loss": mll_loss_value,
                    "surrogate/dyhpo/training_power_law_loss": power_law_loss_value,
                    "surrogate/dyhpo/training_MAE": mae_value,
                    "surrogate/dyhpo/training_power_law_MAE": power_law_mae_value,
                    "surrogate/dyhpo/training_lengthscale": lengthscale_value,
                    "surrogate/dyhpo/training_noise": noise_value,
                    "surrogate/dyhpo/epoch": DyHPOModel._global_epoch,
                    "surrogate/dyhpo/training_errors": DyHPOModel._training_errors,
                    "surrogate/dyhpo/learning_rate": current_lr,
                    "surrogate/dyhpo/nan_gradients": DyHPOModel._nan_gradients,
                }
                if self.l1_loss_factor != 0:
                    wandb_data["surrogate/dyhpo/l1_loss"] = l1_loss_value
                if val_dataset:
                    wandb_data["surrogate/check_training/epoch"] = DyHPOModel._global_epoch
                    wandb_data["surrogate/check_training/train_loss"] = mae_value
                    wandb_data["surrogate/check_training/validation_loss"] = val_mae_value
                    wandb_data["surrogate/check_training/validation_power_law_MAE"] = val_power_law_mae
                    wandb_data["surrogate/check_training/mean_correlation"] = mean_correlation
                    wandb_data["surrogate/check_training/acq_correlation"] = acq_correlation
                    wandb_data["surrogate/check_training/validation_crps"] = crps_score
                    wandb_data["surrogate/check_training/validation_std_correlation"] = val_correlation_value
                    wandb_data["surrogate/check_training/validation_std_coverage_1sigma"] = coverage_1std
                    wandb_data["surrogate/check_training/validation_std_coverage_2sigma"] = coverage_2std
                    wandb_data["surrogate/check_training/validation_std_coverage_3sigma"] = coverage_3std
                if gv.PLOT_SUGGEST_LR or self.meta.use_suggested_learning_rate:
                    wandb_data["surrogate/dyhpo/suggested_lr"] = DyHPOModel._suggested_lr

                wandb.log(wandb_data)

            except Exception as training_error:
                self.logger.error(f'The following error happened at epoch {epoch_nr} while training: {training_error}')
                self.logger.exception(training_error)
                traceback.print_exc()
                # raise training_error
                # An error has happened, trigger the restart of the optimization and restart
                # the model with default hyperparameters.
                training_errored = True
                DyHPOModel._training_errors += 1
                wandb.log({
                    f"surrogate/dyhpo/epoch": DyHPOModel._global_epoch,
                    f"surrogate/dyhpo/training_errors": DyHPOModel._training_errors,
                })
                break

        check_seed_torch = torch.random.get_rng_state().sum()
        self.logger.debug(f"end rng_state {check_seed_torch}")

        if training_errored:
            self.save_checkpoint(initial_state)
            self.load_checkpoint()

        is_diverging = False
        # metric too high, time to restart, or we risk divergence
        if self.meta.reset_on_divergence and mae_value > 0.4:
            is_diverging = True

        # return state -1 signals that training has failed. Hyperparameter optimizer decides what to do when training
        # fails. Currently, hyperparameter optimizer stop refining and start training from scratch.
        if training_errored or is_diverging:
            return_state = -1
        elif is_nan_gradient:
            return_state = 2
        else:
            return_state = 0
        return return_state, loss_value

    def predict(
        self,
        test_data: TabularDataset,
        train_data: TabularDataset,
        target_normalization_inverse_fn=None,
        target_normalization_std_inverse_fn=None
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], Optional[Dict[str, Any]]]:
        means, stds, predict_info, power_law_loss = self._predict(
            test_data=test_data,
            train_data=train_data,
            target_normalization_inverse_fn=target_normalization_inverse_fn,
            target_normalization_std_inverse_fn=target_normalization_std_inverse_fn
        )

        power_law_loss = power_law_loss / len(test_data)

        wandb.log({
            "surrogate/dyhpo/testing_power_law_MAE": power_law_loss
        })

        return means, stds, predict_info

    def _predict(
        self,
        test_data: TabularDataset,
        train_data: TabularDataset,
        target_normalization_inverse_fn=None,
        target_normalization_std_inverse_fn=None
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], Optional[Dict[str, Any]], float]:
        """

        Args:
            train_data: A dictionary that has the training
                examples, features, budgets and learning curves.
            test_data: Same as for the training data, but it is
                for the testing part and it does not feature labels.

        Returns:
            means, stds: The means of the predictions for the
                testing points and the standard deviations.
        """
        model_device = next(self.parameters()).device
        train_data.to(model_device)
        test_data.to(model_device)

        self.eval()

        with torch.no_grad():  # gpytorch.settings.fast_pred_var():
            projected_train_x, _ = self.feature_extractor(
                train_data.X,
                train_data.budgets,
                train_data.curves,
            )
            self.model.set_train_data(inputs=projected_train_x, targets=train_data.Y, strict=False)
            projected_test_x, test_predict_infos = self.feature_extractor(
                test_data.X,
                test_data.budgets,
                test_data.curves,
            )
            preds: gpytorch.distributions.Distribution = self.likelihood(self.model(projected_test_x))

            means = preds.mean
            stds = preds.stddev

            power_law_output = projected_test_x[:, -1]

            if target_normalization_inverse_fn:
                means = target_normalization_inverse_fn(means)
                stds = target_normalization_std_inverse_fn(stds)
                power_law_output = target_normalization_inverse_fn(power_law_output)

            power_law_loss = F.l1_loss(means, power_law_output, reduction='sum')
            power_law_loss_value = power_law_loss.detach().to('cpu').item()

        means = means.detach().to('cpu').numpy().reshape(-1, )
        stds = stds.detach().to('cpu').numpy().reshape(-1, )

        predict_infos = test_predict_infos
        if predict_infos is not None:
            predict_infos = {key: value.detach().to('cpu').numpy() for key, value in predict_infos.items()}

        return means, stds, predict_infos, power_law_loss_value

    def _batch_predict(
        self,
        test_data: TabularDataset,
        train_data: TabularDataset,
        **kwargs
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], Optional[Dict[str, Any]], float]:
        """

        Args:
            train_data: A dictionary that has the training
                examples, features, budgets and learning curves.
            test_data: Same as for the training data, but it is
                for the testing part and it does not feature labels.

        Returns:
            means, stds: The means of the predictions for the
                testing points and the standard deviations.
        """
        model_device = next(self.parameters()).device
        train_data.to(model_device)
        test_data.to(model_device)

        self.eval()

        batch_size = 512
        last_batch_index = int(len(test_data) / batch_size) - 1 if len(test_data) >= batch_size else 0

        mean_data = []
        std_data = []
        predict_info_data = []
        power_law_loss_data = []

        for i in range(0, last_batch_index + 1):
            if i == last_batch_index:
                batch_indices = range(i * batch_size, len(test_data))
            else:
                batch_indices = range(i * batch_size, ((i + 1) * batch_size))
            batch_data = test_data.get_subset(batch_indices)
            means, stds, predict_info, power_law_loss = \
                self._predict(test_data=batch_data, train_data=train_data, **kwargs)

            mean_data.append(means)
            std_data.append(stds)
            predict_info_data.append(predict_info)
            power_law_loss_data.append(power_law_loss)

        power_law_loss = np.sum(power_law_loss_data).item() / len(test_data)
        means = np.concatenate(mean_data, axis=0)
        stds = np.concatenate(std_data, axis=0)

        predict_infos = None
        if predict_info_data[0] is not None:
            predict_infos = \
                {key: np.concatenate([d[key] for d in predict_info_data]) for key in predict_info_data[0].keys()}

        return means, stds, predict_infos, power_law_loss

    def load_checkpoint(self):
        """
        Load the state from a previous checkpoint.
        """
        checkpoint = torch.load(self.checkpoint_file)
        self.model.load_state_dict(checkpoint['gp_state_dict'])
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.likelihood.load_state_dict(checkpoint['likelihood_state_dict'])

    def save_checkpoint(self, state: Dict = None):
        """
        Save the given state or the current state in a
        checkpoint file.

        Args:
            state: The state to save, if none, it will
            save the current state.
        """

        if state is None:
            torch.save(
                self.get_state(),
                self.checkpoint_file,
            )
        else:
            torch.save(
                state,
                self.checkpoint_file,
            )

    def get_state(self) -> Dict[str, Dict]:
        """
        Get the current state of the surrogate.

        Returns:
            current_state: A dictionary that represents
                the current state of the surrogate model.
        """
        current_state = {
            'gp_state_dict': deepcopy(self.model.state_dict()),
            'feature_extractor_state_dict': deepcopy(self.feature_extractor.state_dict()),
            'likelihood_state_dict': deepcopy(self.likelihood.state_dict()),
        }

        return current_state

    def to(self, dev):
        self.feature_extractor.to(dev)
        self.model.to(dev)
        self.likelihood.to(dev)

    def train(self, **kwargs):
        self.feature_extractor.train(**kwargs)
        self.model.train(**kwargs)
        self.likelihood.train(**kwargs)

    def eval(self):
        self.feature_extractor.eval()
        self.model.eval()
        self.likelihood.eval()

    @classproperty
    def meta_use_learning_curve(cls):
        model_class = get_class("src/models/deep_kernel_learning", cls.meta.feature_class_name)
        return model_class.meta_use_learning_curve

    @classproperty
    def meta_use_learning_curve_mask(cls):
        model_class = get_class("src/models/deep_kernel_learning", cls.meta.feature_class_name)
        return model_class.meta_use_learning_curve_mask

    @classproperty
    def meta_output_act_func(cls):
        model_class = get_class("src/models/deep_kernel_learning", cls.meta.feature_class_name)
        return model_class.meta_output_act_func

    @classproperty
    def meta_cnn_kernel_size(cls):
        model_class = get_class("src/models/deep_kernel_learning", cls.meta.feature_class_name)
        return model_class.meta_cnn_kernel_size

    def get_feature_extractor_output_size(self, nr_features):
        valid_budget_size = 50
        # Create a dummy input with the appropriate input size
        dummy_x_input = torch.zeros(4, nr_features)
        dummy_budget_input = torch.ones(4)
        dummy_learning_curves_input = torch.zeros(4, valid_budget_size)
        self.feature_extractor.eval()
        output, _ = self.feature_extractor(dummy_x_input, dummy_budget_input, dummy_learning_curves_input)
        output_size = output.shape[-1]
        return output_size

    def set_target_normalization_inverse_function(self, fn, std_fn=None):
        self.target_normalization_inverse_fn = fn
        self.target_normalization_std_inverse_fn = std_fn

    def suggest_learning_rate(self, train_dataloader=None):
        model_device = next(self.parameters()).device
        model = deepcopy(self)
        model.hook_remove()
        model.set_optimizer(use_scheduler=False)
        lr_finder = LRFinder(model=model, optimizer=model.optimizer, criterion=None, device=model_device,
                             is_used=self.meta.use_suggested_learning_rate, is_dyhpo=True)
        lr_finder.range_test(train_dataloader, start_lr=1e-8, end_lr=1, num_iter=100, diverge_th=1.1)
        # lr_finder.plot()
        suggested_lr = lr_finder.get_suggested_lr()
        # print(f"suggested_lr {suggested_lr}")
        return suggested_lr

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove unpickable objects from the state dictionary
        state['logger'] = None
        state['_modules']['feature_extractor'].logger = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore unpickable objects from the state dictionary
        self.logger = logger
        self.feature_extractor.logger = logger

    def hook_remove(self):
        self.feature_extractor.hook_remove()

    @property
    def has_batchnorm_layers(self):
        return self.feature_extractor.has_batchnorm_layers

    def reset(self):
        if gv.IS_WANDB and gv.PLOT_GRADIENTS:
            wandb.unwatch()
