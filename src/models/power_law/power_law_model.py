import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
import wandb
from abc import ABC, abstractmethod
from functools import partial
from typing import List, Tuple, Dict, Optional, Any, Union, Type
import inspect

from src.models.base.base_pytorch_module import BasePytorchModule
import src.models.activation_functions
from src.utils.utils import get_class_from_package, get_class_from_packages
import global_variables as gv


class PowerLawModel(BasePytorchModule, ABC):
    _instance_counter = 0
    _global_epoch = {}
    _clip_gradients = 0
    _view_gradient_step = 0

    # gradients_tracking_pd: Optional[pd.DataFrame] = None

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

        self.linear_net = self.get_linear_net()

        if hasattr(self.meta, "use_learning_curve") and self.meta.use_learning_curve:
            self.cnn_net = self.get_cnn_net()

        self.criterion = get_class_from_package(torch.nn, self.meta.loss_function)()

        self.has_batchnorm_layers = False
        self.optimizer = None
        self.lr_scheduler = None

        if gv.PLOT_GRADIENTS:
            if hasattr(self, 'param_names'):
                hook = partial(PowerLawModel.gradient_logging_hook, names=self.param_names)
                self.linear_net.register_full_backward_hook(hook=hook)

        if hasattr(self.meta, 'clip_gradients') and self.meta.clip_gradients != 0:
            PowerLawModel._clip_gradients = self.meta.clip_gradients
            for layer in self.linear_net:
                layer.register_full_backward_hook(PowerLawModel.gradient_clipping_hook)

        self.post_init()

    @staticmethod
    def gradient_clipping_hook(module, grad_input, grad_output):
        for grad in grad_input:
            if grad is not None:
                grad.clamp_(-PowerLawModel._clip_gradients, PowerLawModel._clip_gradients)
        return grad_input

    def post_init(self):
        self.has_batchnorm_layers = self.get_has_batchnorm_layers()
        self.has_batchnorm_layers = True

        # self.set_optimizer()

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

    def set_optimizer(self, **kwargs):
        optimizer_class = get_class_from_package(torch.optim, self.meta.optimizer)
        self.optimizer = optimizer_class(self.parameters(), lr=self.meta.learning_rate)

        if hasattr(self.meta, 'learning_rate_scheduler') and self.meta.learning_rate_scheduler:
            lr_scheduler_class = get_class_from_package(torch.optim.lr_scheduler, self.meta.learning_rate_scheduler)
            args = {}
            if hasattr(self.meta, 'learning_rate_scheduler_args') and self.meta.learning_rate_scheduler_args:
                args = self.meta.learning_rate_scheduler_args

            epochs = kwargs['epochs']
            if "total_iters_factor" in args:
                args["total_iters"] = epochs * args["total_iters_factor"]
                args["T_max"] = epochs * args["total_iters_factor"]

            args["verbose"] = True
            args["epochs"] = epochs
            args["total_steps"] = epochs
            args["lr_lambda"] = lambda step: 1 - step / args["total_iters"]

            arg_names = inspect.getfullargspec(lr_scheduler_class.__init__).args
            relevant_args = {k: v for k, v in args.items() if k in arg_names}
            self.lr_scheduler = lr_scheduler_class(self.optimizer, **relevant_args)

    def training_step(self):
        batch = next(self.train_dataloader_it)
        batch_examples, batch_labels, batch_budgets, batch_curves = batch
        nr_examples_batch = batch_examples.shape[0]
        # if only one example in the batch, skip the batch.
        # Otherwise, the code will fail because of batchnormalization.
        if self.has_batchnorm_layers and nr_examples_batch == 1:
            return 0

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
        self.optimizer.step()
        return loss.item()

    def train_epoch(self):
        running_loss = 0
        while True:
            try:
                loss = self.training_step()
                running_loss += loss
            except StopIteration:
                self.train_dataloader_it = iter(self.train_dataloader)
                break
        normalized_loss = running_loss / len(self.train_dataloader)
        return normalized_loss

    def train_loop(self, nr_epochs, train_dataloader=None, reset_optimizer=False):
        self.set_dataloader(train_dataloader)

        if reset_optimizer or self.optimizer is None:
            self.set_optimizer(epochs=nr_epochs)

        self.set_seed(self.seed)
        self.train()

        patience_rounds = 0
        best_loss = np.inf
        best_state = deepcopy(self.state_dict())

        for epoch in range(0, nr_epochs):
            normalized_loss = self.train_epoch()
            self.logger.debug(f'Epoch {epoch + 1}, Loss:{normalized_loss}')
            PowerLawModel._global_epoch[self.instance_id] += 1

            if self.lr_scheduler and self.optimizer._step_count > 0:
                self.lr_scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            wandb_data = {
                f"surrogate/model_{self.instance_id}/training_loss": normalized_loss,
                f"surrogate/model_{self.instance_id}/epoch": PowerLawModel._global_epoch[self.instance_id],
                f"surrogate/model_{self.instance_id}/learning_rate": current_lr
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

    def __del__(self):
        PowerLawModel._instance_counter = 0

    def set_register_full_backward_hook(self, **kwargs):
        if hasattr(self, 'param_names'):
            kwargs['hook'] = partial(kwargs['hook'], names=self.param_names)
            self.linear_net.register_full_backward_hook(**kwargs)

    @staticmethod
    def gradient_logging_hook(module, grad_input, grad_output, names):
        PowerLawModel._view_gradient_step += 1

        grads = grad_output[0]
        grads = torch.abs(grads).mean(dim=0)
        grads = grads.numpy()

        # if HyperparameterOptimizer.gradients_tracking_pd is None:
        #     HyperparameterOptimizer.gradients_tracking_pd = pd.DataFrame(columns=names)
        #
        # length = len(HyperparameterOptimizer.gradients_tracking_pd)
        # HyperparameterOptimizer.gradients_tracking_pd.loc[length] = grads

        if gv.IS_WANDB:
            wandb_data = {
                'gradients/gradient_step': PowerLawModel._view_gradient_step
            }
            for i, name in enumerate(names):
                wandb_data[f'gradients/{name}'] = grads[i]
            wandb.log(wandb_data)

        # if HyperparameterOptimizer._view_gradient_step % 1000 == 0 and \
        #     HyperparameterOptimizer._view_gradient_step > 1:
        #     plot_path = os.path.join(
        #         'gradients',
        #         f'gradient_plots_{HyperparameterOptimizer._view_gradient_step}',
        #     )
        #
        #     plot_line(ydata=HyperparameterOptimizer.gradients_tracking_pd, x_label='training step', y_label='gradient',
        #               title=f"Gradients at training step {HyperparameterOptimizer._view_gradient_step}",
        #               path=plot_path)
