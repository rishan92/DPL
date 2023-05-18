import math

import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
import wandb
from abc import ABC, abstractmethod
from functools import partial
from typing import List, Tuple, Dict, Optional, Any, Union, Type
import inspect
from torch.nn.utils import clip_grad_norm_
import warnings
from loguru import logger
import torch.nn.functional as F
from scipy.stats import spearmanr
import properscoring as ps

from src.models.base.base_pytorch_module import BasePytorchModule
import src.models.activation_functions
from src.utils.utils import get_class_from_package, get_class_from_packages, get_inverse_function_class
import global_variables as gv
from src.utils.torch_lr_finder import LRFinder
from src.utils.utils import acq


class PowerLawModel(BasePytorchModule, ABC):
    _instance_counter = 0
    _global_epoch = {}
    _suggested_lr = None
    _nan_gradients = 0
    _validation_outputs = {}
    _validation_epoch_outputs = {}
    _validation_corr = {}
    _validation_online = {}

    def __init__(
        self,
        nr_features,
        max_instances,
        seed=None,
        checkpoint_path='.'
    ):
        """
        Args:
            nr_initial_features: int
                The number of features per example.
            nr_units: int
                The number of units for every layer.
            nr_layers: int
                The number of layers for the neural network.
            use_learning_curve: bool
                If the learning curve should be use in the network.
            kernel_size: int
                The size of the kernel that is applied in the cnn layer.
            nr_filters: int
                The number of filters that are used in the cnn layers.
            nr_cnn_layers: int
                The number of cnn layers to be used.
        """
        super().__init__(nr_features=nr_features, seed=seed, checkpoint_path=checkpoint_path)
        self.max_instances = max_instances
        self.instance_id = PowerLawModel._instance_counter
        PowerLawModel._instance_counter += 1
        PowerLawModel._instance_counter %= self.max_instances

        if self.instance_id not in PowerLawModel._global_epoch:
            PowerLawModel._global_epoch[self.instance_id] = 0

        self.act_func = None
        self.last_act_func = None
        self.alpha_act_func = None
        self.beta_act_func = None
        self.gamma_act_func = None
        self.alphai_act_func = None
        self.betai_act_func = None
        self.gammai_act_func = None
        self.output_act_func = None
        self.output_act_inverse_func = None
        self.linear_net = None
        self.cnn_net = None

        if hasattr(self.meta, "act_func"):
            self.act_func = get_class_from_packages([torch.nn, src.models.activation_functions], self.meta.act_func)()
        if hasattr(self.meta, "last_act_func"):
            self.last_act_func = get_class_from_packages([torch.nn, src.models.activation_functions],
                                                         self.meta.last_act_func)()
        if hasattr(self.meta, "alpha_act_func"):
            self.alpha_act_func = get_class_from_packages([torch.nn, src.models.activation_functions],
                                                          self.meta.alpha_act_func)()
        if hasattr(self.meta, "beta_act_func"):
            self.beta_act_func = get_class_from_packages([torch.nn, src.models.activation_functions],
                                                         self.meta.beta_act_func)()
        if hasattr(self.meta, "gamma_act_func"):
            self.gamma_act_func = get_class_from_packages([torch.nn, src.models.activation_functions],
                                                          self.meta.gamma_act_func)()
        if hasattr(self.meta, "output_act_func") and self.meta.output_act_func:
            self.output_act_func = get_class_from_packages([torch.nn, src.models.activation_functions],
                                                           self.meta.output_act_func)()

            output_act_inverse_class = get_inverse_function_class(self.meta.output_act_func)
            self.output_act_inverse_func = output_act_inverse_class() if output_act_inverse_class else None

            # val = [-1000, -10, -2, -1, 0, 0.25, 0.5, 0.75, 1, 2, 10, 1000]
            # a = self.output_act_func(torch.tensor(val))
            # b = self.output_act_inverse_func(a)

        if hasattr(self.meta, "alphai_act_func"):
            self.alphai_act_func = get_class_from_packages([torch.nn, src.models.activation_functions],
                                                           self.meta.alphai_act_func)()
        if hasattr(self.meta, "betai_act_func"):
            self.betai_act_func = get_class_from_packages([torch.nn, src.models.activation_functions],
                                                          self.meta.betai_act_func)()
        if hasattr(self.meta, "gammai_act_func"):
            self.gammai_act_func = get_class_from_packages([torch.nn, src.models.activation_functions],
                                                           self.meta.gammai_act_func)()

        self.linear_net = self.get_linear_net()

        if hasattr(self.meta, "use_learning_curve") and self.meta.use_learning_curve:
            self.cnn_net = self.get_cnn_net()

        if self.meta.use_sample_weights or self.meta.use_sample_weight_by_budget or self.meta.use_sample_weight_by_label:
            reduction = 'none'
        else:
            reduction = 'mean'
        self.criterion = get_class_from_package(torch.nn, self.meta.loss_function)(reduction=reduction)

        self.has_batchnorm_layers = False
        self.optimizer = None
        self.lr_scheduler = None
        self.target_normalization_inverse_fn = None

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

        self.hook_handle = None

        self.clip_gradients_value = None
        if hasattr(self.meta, 'clip_gradients') and self.meta.clip_gradients != 0:
            self.clip_gradients_value = self.meta.clip_gradients

        PowerLawModel._validation_online['mean'] = []
        PowerLawModel._validation_online['std'] = []

        self.post_init()

    def post_init(self):
        self.has_batchnorm_layers = self.get_has_batchnorm_layers()
        self.has_batchnorm_layers = True

        if gv.PLOT_GRADIENTS and self.instance_id == 0:
            if hasattr(self, 'param_names'):
                hook = partial(self.gradient_logging_hook, names=self.param_names)
                if hasattr(self, 'linear_net') and self.linear_net is not None:
                    self.hook_handle = self.linear_net.register_full_backward_hook(hook=hook)
                else:
                    warnings.warn("Gradient flow tracking with wandb is not supported for this module.")

        if gv.IS_WANDB and gv.PLOT_GRADIENTS and self.instance_id == 0:
            wandb.watch(self, log='all', idx=0, log_freq=10)

    def get_has_batchnorm_layers(self):
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                return True
        return False

    def get_linear_net(self):
        raise NotImplementedError

    def get_cnn_net(self):
        raise NotImplementedError

    def forward(self, batch):
        raise NotImplementedError

    def set_optimizer(self, use_scheduler=False, **kwargs):
        is_refine = self.optimizer is not None
        if is_refine and hasattr(self.meta, 'refine_learning_rate') and self.meta.refine_learning_rate:
            learning_rate = self.meta.refine_learning_rate
        else:
            learning_rate = self.meta.learning_rate
        optimizer_class = get_class_from_package(torch.optim, self.meta.optimizer)
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

    def train_epoch(self, max_batches=None):
        running_loss = 0
        is_nan_gradient = False
        batch_count = 0
        while True:
            try:
                batch = next(self.train_dataloader_it)
                batch_examples, batch_labels, batch_budgets, batch_curves, batch_weights = batch
                nr_examples_batch = batch_examples.shape[0]
                # if only one example in the batch, skip the batch.
                # Otherwise, the code will fail because of batchnormalization.
                if self.has_batchnorm_layers and nr_examples_batch == 1:
                    continue

                # zero the parameter gradients
                self.optimizer.zero_grad(set_to_none=True)
                outputs, predict_info = self((batch_examples, batch_budgets, batch_curves))
                # if outputs.is_complex():
                #     imag_loss_factor = 1
                #     imag_loss = torch.abs(outputs.imag).mean()
                #     loss = self.criterion(outputs.real, batch_labels) + imag_loss_factor * imag_loss
                # else:
                #     loss = self.criterion(outputs, batch_labels)

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

                if self.output_constraint_factor != 0:
                    lower_loss = torch.clamp(-1 * outputs, min=0)
                    upper_loss = torch.clamp(outputs - 1, min=0)
                    output_constraint_loss = torch.mean(lower_loss + upper_loss)
                else:
                    output_constraint_loss = torch.tensor(0.0, requires_grad=True)

                # if self.meta.use_sample_weights:
                #     batch_labels = batch_labels + batch_noise

                criterion_loss = self.criterion(outputs, batch_labels)

                if self.meta.use_sample_weights:
                    batch_weights /= batch_weights.sum()
                    batch_weights *= nr_examples_batch
                    criterion_loss = (criterion_loss * batch_weights).mean()
                elif self.meta.use_sample_weight_by_budget or self.meta.use_sample_weight_by_label:
                    criterion_loss = (criterion_loss * batch_weights).mean()

                loss = criterion_loss + self.regularization_factor * l1_norm + \
                       self.alpha_beta_constraint_factor * alpha_beta_constraint_loss + \
                       self.gamma_constraint_factor * gamma_constraint_loss + \
                       self.output_constraint_factor * output_constraint_loss

                loss.backward()

                if not is_nan_gradient:
                    for name, param in self.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any():
                                is_nan_gradient = True
                                break

                if self.clip_gradients_value:
                    clip_grad_norm_(self.parameters(), self.clip_gradients_value)

                self.optimizer.step()

                running_loss += loss.item() * nr_examples_batch
                batch_count += 1

                if max_batches is not None and batch_count >= max_batches:
                    break
            except StopIteration:
                self.train_dataloader_it = iter(self.train_dataloader)
                break
        if max_batches is None:
            normalized_loss = running_loss / len(self.train_dataloader.dataset)
        else:
            normalized_loss = running_loss
        return normalized_loss, is_nan_gradient

    def validation_epoch(self, epoch):
        self.eval()
        running_loss = 0
        predict_infos = []
        iteration = 0
        batch_size = 512

        while True:
            try:
                batch = next(self.val_dataloader_it)
                batch_examples, batch_labels, batch_budgets, batch_curves, _ = batch
                nr_examples_batch = batch_examples.shape[0]
                # if only one example in the batch, skip the batch.
                # Otherwise, the code will fail because of batchnormalization.
                if self.has_batchnorm_layers and nr_examples_batch == 1:
                    continue

                outputs, predict_info = self((batch_examples, batch_budgets, batch_curves))
                # if outputs.is_complex():
                #     imag_loss_factor = 1
                #     imag_loss = torch.abs(outputs.imag).mean()
                #     loss = self.criterion(outputs.real, batch_labels) + imag_loss_factor * imag_loss
                # else:
                #     loss = self.criterion(outputs, batch_labels)
                predict_infos.append(predict_info)

                if self.target_normalization_inverse_fn:
                    outputs = self.target_normalization_inverse_fn(outputs)

                loss = F.l1_loss(outputs, batch_labels, reduction='sum')

                running_loss += loss.item()

                new_value = outputs.detach()
                start_index = iteration * batch_size
                end_index = iteration * batch_size + nr_examples_batch
                if self.instance_id == 0:
                    PowerLawModel._validation_online['mean'][epoch, start_index:end_index] = new_value
                else:
                    delta = new_value - PowerLawModel._validation_online['mean'][epoch, start_index:end_index]
                    PowerLawModel._validation_online['mean'][epoch, start_index:end_index] += delta / (
                        self.instance_id + 1)
                    delta2 = new_value - PowerLawModel._validation_online['mean'][epoch, start_index:end_index]
                    PowerLawModel._validation_online['std'][epoch, start_index:end_index] += delta * delta2

                iteration += 1
            except StopIteration:
                self.val_dataloader_it = iter(self.val_dataloader)
                break
        normalized_loss = running_loss / len(self.val_dataloader.dataset)

        return normalized_loss

    def train_loop(self, nr_epochs, train_dataloader=None, reset_optimizer=False, val_dataloader=None,
                   is_lr_finder=False):
        if (gv.PLOT_SUGGEST_LR or self.meta.use_suggested_learning_rate) and self.instance_id == 0 and not is_lr_finder:
            PowerLawModel._suggested_lr = self.suggest_learning_rate(train_dataloader=train_dataloader)
            # if PowerLawModel._suggested_lr is not None:
            #     print(f"Suggested LR: {PowerLawModel._suggested_lr:.2E}")

        self.set_dataloader(train_dataloader=train_dataloader, val_dataloader=val_dataloader)

        if reset_optimizer or self.optimizer is None:
            self.set_optimizer(epochs=nr_epochs, use_scheduler=True)

        if self.meta.use_suggested_learning_rate and PowerLawModel._suggested_lr is not None and not is_lr_finder:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = PowerLawModel._suggested_lr

        self.set_seed(self.seed)
        self.train()

        patience_rounds = 0
        best_loss = np.PINF
        best_state = deepcopy(self.state_dict())
        normalized_val_loss = None
        is_nan_gradient = False
        normalized_loss = 0
        max_batches = 1 if is_lr_finder else None

        for epoch in range(0, nr_epochs):
            normalized_loss, is_nan_gradient = self.train_epoch(max_batches=max_batches)
            self.logger.debug(f'Epoch {epoch + 1}, Loss:{normalized_loss}')
            if not is_lr_finder:
                PowerLawModel._global_epoch[self.instance_id] += 1

            if self.lr_scheduler:
                if self.meta.learning_rate_scheduler != "ReduceLROnPlateau":
                    if self.optimizer._step_count > 0:
                        self.lr_scheduler.step()
                else:
                    self.lr_scheduler.step(normalized_loss)

            if val_dataloader:
                if self.instance_id == 0 and epoch == 0:
                    PowerLawModel._validation_online['mean'] = torch.zeros(
                        (nr_epochs, len(self.val_dataloader.dataset)))
                    PowerLawModel._validation_online['std'] = torch.zeros((nr_epochs, len(self.val_dataloader.dataset)))
                normalized_val_loss = self.validation_epoch(epoch)
                self.train()

            if is_nan_gradient and not is_lr_finder:
                PowerLawModel._nan_gradients += 1

            # print(f"lr {self.optimizer.param_groups[0]['lr']}")

            if self.instance_id == 0 and not is_lr_finder:
                current_lr = self.optimizer.param_groups[0]['lr']
                wandb_data = {
                    f"surrogate/model_{self.instance_id}/training_loss": normalized_loss,
                    f"surrogate/model_{self.instance_id}/epoch": PowerLawModel._global_epoch[self.instance_id],
                    f"surrogate/model_{self.instance_id}/learning_rate": current_lr,
                    f"surrogate/model_{self.instance_id}/nan_gradients": PowerLawModel._nan_gradients
                }
                if val_dataloader:
                    wandb_data[f"surrogate/model_{self.instance_id}/validation_loss"] = normalized_val_loss

                if gv.PLOT_SUGGEST_LR or self.meta.use_suggested_learning_rate:
                    wandb_data[f"surrogate/model_{self.instance_id}/suggested_lr"] = PowerLawModel._suggested_lr
                wandb.log(wandb_data)

            if self.instance_id == self.max_instances - 1 and not is_lr_finder:
                if val_dataloader:
                    labels = val_dataloader.dataset.Y
                    means = PowerLawModel._validation_online['mean'][epoch, :]
                    stds = PowerLawModel._validation_online['std'][epoch, :]
                    stds = (stds / (self.max_instances - 1)) ** 0.5
                    normalized_val_loss = F.l1_loss(means, labels, reduction='mean')

                    means = means.detach().numpy()
                    stds = stds.detach().numpy()
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

                    wandb_data = {
                        f"surrogate/check_training/epoch": PowerLawModel._global_epoch[self.instance_id],
                        f"surrogate/check_training/train_loss": normalized_loss,
                        f"surrogate/check_training/validation_loss": normalized_val_loss,
                        f"surrogate/check_training/mean_correlation": mean_correlation,
                        f"surrogate/check_training/acq_correlation": acq_correlation,
                        f"surrogate/check_training/validation_crps": crps_score,
                        f"surrogate/check_training/validation_std_correlation": val_correlation_value,
                        f"surrogate/check_training/validation_std_coverage_1sigma": coverage_1std,
                        f"surrogate/check_training/validation_std_coverage_2sigma": coverage_2std,
                        f"surrogate/check_training/validation_std_coverage_3sigma": coverage_3std,
                    }
                    wandb.log(wandb_data)

            if self.meta.activate_early_stopping:
                if normalized_loss < best_loss:
                    best_state = deepcopy(self.state_dict())
                    best_loss = normalized_loss
                    patience_rounds = 0
                elif normalized_loss > best_loss:
                    patience_rounds += 1
                    if patience_rounds == self.meta.early_stopping_it:
                        self.load_state_dict(best_state)
                        self.logger.info(f'Stopping training since validation loss is not improving')
                        break

        check_seed_torch = torch.random.get_rng_state().sum()
        self.logger.debug(f"end rng_state {check_seed_torch}")

        if self.meta.activate_early_stopping:
            self.load_state_dict(best_state)

        return_state = 0
        if is_nan_gradient:
            return_state = 2

        return return_state, normalized_loss

    def predict(self, test_data):
        self.eval()
        predictions, predict_infos = self((test_data.X, test_data.budgets, test_data.curves))
        # if predictions.is_complex():
        #     predictions = predictions.real
        return predictions, predict_infos

    @staticmethod
    def get_optimizer_step_count(optimizer):
        for group in optimizer.param_groups:
            for p in group['params']:
                if 'step' in optimizer.state[p]:
                    return optimizer.state[p]['step']
        return 0

    def suggest_learning_rate(self, train_dataloader=None):
        model_device = next(self.parameters()).device
        model = deepcopy(self)
        # Change instance id from zero to stop recursive calls.
        model.instance_id = 1000
        model.hook_remove()
        model.set_optimizer(use_scheduler=False)
        lr_finder = LRFinder(model=model, optimizer=model.optimizer, criterion=model.criterion, device=model_device,
                             is_used=self.meta.use_suggested_learning_rate, is_dyhpo=False)
        lr_finder.range_test(train_dataloader, start_lr=1e-8, end_lr=1, num_iter=100, diverge_th=1.1)
        # lr_finder.plot()
        suggested_lr = lr_finder.get_suggested_lr()
        # print(f"suggested_lr {suggested_lr}")
        return suggested_lr

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove unpickable objects from the state dictionary
        state['logger'] = None
        state['train_dataloader'] = None
        state['train_dataloader_it'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Restore unpickable objects from the state dictionary
        self.logger = logger

    def set_target_normalization_inverse_function(self, fn, std_fn=None):
        self.target_normalization_inverse_fn = fn

    def hook_remove(self):
        if self.hook_handle:
            self.hook_handle.remove()

    def reset(self):
        PowerLawModel._instance_counter = 0
