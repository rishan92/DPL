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

from src.models.base.base_pytorch_module import BasePytorchModule
import src.models.activation_functions
from src.utils.utils import get_class_from_package, get_class_from_packages, get_inverse_function_class
import global_variables as gv
from src.utils.torch_lr_finder import LRFinder


class PowerLawModel(BasePytorchModule, ABC):
    _instance_counter = 0
    _global_epoch = {}
    _suggested_lr = None
    _nan_gradients = 0

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

        self.criterion = get_class_from_package(torch.nn, self.meta.loss_function)()

        self.has_batchnorm_layers = False
        self.optimizer = None
        self.lr_scheduler = None

        self.hook_handle = None

        self.clip_gradients_value = None
        if hasattr(self.meta, 'clip_gradients') and self.meta.clip_gradients != 0:
            self.clip_gradients_value = self.meta.clip_gradients

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
            if is_refine and 'refine_max_lr' in args and args["refine_max_lr"]:
                args["max_lr"] = args["refine_max_lr"]

            arg_names = inspect.getfullargspec(lr_scheduler_class.__init__).args
            relevant_args = {k: v for k, v in args.items() if k in arg_names}
            self.lr_scheduler = lr_scheduler_class(self.optimizer, **relevant_args)

    def train_epoch(self):
        running_loss = 0
        is_nan_gradient = False
        while True:
            try:
                batch = next(self.train_dataloader_it)
                batch_examples, batch_labels, batch_budgets, batch_curves = batch
                nr_examples_batch = batch_examples.shape[0]
                # if only one example in the batch, skip the batch.
                # Otherwise, the code will fail because of batchnormalization.
                if self.has_batchnorm_layers and nr_examples_batch == 1:
                    return 0, is_nan_gradient

                # zero the parameter gradients
                self.optimizer.zero_grad(set_to_none=True)
                outputs, _ = self((batch_examples, batch_budgets, batch_curves))
                # if outputs.is_complex():
                #     imag_loss_factor = 1
                #     imag_loss = torch.abs(outputs.imag).mean()
                #     loss = self.criterion(outputs.real, batch_labels) + imag_loss_factor * imag_loss
                # else:
                #     loss = self.criterion(outputs, batch_labels)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()

                if self.clip_gradients_value:
                    clip_grad_norm_(self.parameters(), self.clip_gradients_value)

                if not is_nan_gradient:
                    for name, param in self.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any():
                                is_nan_gradient = True
                                break

                self.optimizer.step()

                running_loss += loss.item()
            except StopIteration:
                self.train_dataloader_it = iter(self.train_dataloader)
                break
        normalized_loss = running_loss / len(self.train_dataloader)
        return normalized_loss, is_nan_gradient

    def train_loop(self, nr_epochs, train_dataloader=None, reset_optimizer=False):
        if (gv.PLOT_SUGGEST_LR or self.meta.use_suggested_learning_rate) and self.instance_id == 0:
            PowerLawModel._suggested_lr = self.suggest_learning_rate(train_dataloader=train_dataloader)
            # if PowerLawModel._suggested_lr is not None:
            #     print(f"Suggested LR: {PowerLawModel._suggested_lr:.2E}")

        self.set_dataloader(train_dataloader)

        if reset_optimizer or self.optimizer is None:
            self.set_optimizer(epochs=nr_epochs, use_scheduler=True)

        if self.meta.use_suggested_learning_rate and PowerLawModel._suggested_lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = PowerLawModel._suggested_lr

        self.set_seed(self.seed)
        self.train()

        patience_rounds = 0
        best_loss = np.PINF
        best_state = deepcopy(self.state_dict())

        for epoch in range(0, nr_epochs):
            normalized_loss, is_nan_gradient = self.train_epoch()
            self.logger.debug(f'Epoch {epoch + 1}, Loss:{normalized_loss}')
            PowerLawModel._global_epoch[self.instance_id] += 1

            if self.lr_scheduler and self.optimizer._step_count > 0:
                self.lr_scheduler.step()

            if self.instance_id == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                if is_nan_gradient:
                    PowerLawModel._nan_gradients += 1
                wandb_data = {
                    f"surrogate/model_{self.instance_id}/training_loss": normalized_loss,
                    f"surrogate/model_{self.instance_id}/epoch": PowerLawModel._global_epoch[self.instance_id],
                    f"surrogate/model_{self.instance_id}/learning_rate": current_lr,
                    f"surrogate/model_{self.instance_id}/nan_gradients": PowerLawModel._nan_gradients
                }
                if gv.PLOT_SUGGEST_LR or self.meta.use_suggested_learning_rate:
                    wandb_data[f"surrogate/model_{self.instance_id}/suggested_lr"] = PowerLawModel._suggested_lr
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

        return 0

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
                             is_used=self.meta.use_suggested_learning_rate)
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

    def hook_remove(self):
        if self.hook_handle:
            self.hook_handle.remove()

    def reset(self):
        PowerLawModel._instance_counter = 0
