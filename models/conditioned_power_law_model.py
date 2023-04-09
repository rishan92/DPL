import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any
from loguru import logger
from copy import deepcopy
import numpy as np
import random

from models.base_pytorch_module import BasePytorchModule


class ConditionedPowerLawModel(BasePytorchModule):
    _instance_counter = 0

    def __init__(
        self,
        nr_features,
        train_dataloader=None,
        surrogate_configs=None,
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
        super().__init__(nr_features=nr_features, train_dataloader=train_dataloader,
                         surrogate_configs=surrogate_configs)

        self.instance_id = ConditionedPowerLawModel._instance_counter
        ConditionedPowerLawModel._instance_counter += 1

        self.act_func = self.get_class(torch.nn, self.hp.act_func)()
        self.last_act_func = self.get_class(torch.nn, self.hp.last_act_func)()

        self.layers = self.get_linear_net(self.hp)
        self.cnn = self.get_cnn_net(self.hp)

        self.criterion = self.get_class(torch.nn, self.hp.loss_function)()

        self.has_batchnorm_layers = self.get_has_batchnorm_layers()
        self.has_batchnorm_layers = True

        self.optimizer = None
        self.set_optimizer(self.hp)

        self.logger.info(f"Surrogate initialized")

    def get_default_hp(self):
        hp = {
            'nr_units': 128,
            'nr_layers': 2,
            'kernel_size': 3,
            'nr_filters': 4,
            'nr_cnn_layers': 2,
            'use_learning_curve': False,
            'learning_rate': 0.001,
            'act_func': 'LeakyReLU',
            'last_act_func': 'GLU',
            'loss_function': 'L1Loss',
            'optimizer': 'Adam',
            'batch_size': 64,
            'activate_early_stopping': False,
            'early_stopping_it': None,
            'seed': 0,
        }
        return hp

    def get_has_batchnorm_layers(self):
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                return True
        return False

    def get_linear_net(self, hp):
        layers = []
        # adding one since we concatenate the features with the budget
        nr_initial_features = self.nr_features
        if hp.use_learning_curve:
            nr_initial_features = self.nr_features + hp.nr_filters

        act_func = self.get_class(torch.nn, hp.act_func)()
        layers.append(nn.Linear(nr_initial_features, hp.nr_units))
        layers.append(act_func)

        for i in range(2, hp.nr_layers + 1):
            layers.append(nn.Linear(hp.nr_units, hp.nr_units))
            layers.append(act_func)

        last_layer = nn.Linear(hp.nr_units, 3)
        layers.append(last_layer)

        net = torch.nn.Sequential(*layers)
        return net

    def get_cnn_net(self, hp):
        cnn_part = []
        if hp.use_learning_curve:
            act_func = self.get_class(torch.nn, hp.act_func)()
            cnn_part.append(
                nn.Conv1d(
                    in_channels=2,
                    kernel_size=(hp.kernel_size,),
                    out_channels=hp.nr_filters,
                ),
            )
            for i in range(1, hp.nr_cnn_layers):
                cnn_part.append(act_func)
                cnn_part.append(
                    nn.Conv1d(
                        in_channels=hp.nr_filters,
                        kernel_size=(hp.kernel_size,),
                        out_channels=hp.nr_filters,
                    ),
                ),
            cnn_part.append(nn.AdaptiveAvgPool1d(1))

        net = torch.nn.Sequential(*cnn_part)
        return net

    def forward(self, batch):
        """
        Args:
            x: torch.Tensor
                The examples.
            predict_budgets: torch.Tensor
                The budgets for which the performance will be predicted for the
                hyperparameter configurations.
            evaluated_budgets: torch.Tensor
                The budgets for which the hyperparameter configurations have been
                evaluated so far.
            learning_curves: torch.Tensor
                The learning curves for the hyperparameter configurations.
        """
        x, predict_budgets, evaluated_budgets, learning_curves = batch

        # x = torch.cat((x, torch.unsqueeze(evaluated_budgets, 1)), dim=1)
        if self.hp.use_learning_curve:
            lc_features = self.cnn(learning_curves)
            # revert the output from the cnn into nr_rows x nr_kernels.
            lc_features = torch.squeeze(lc_features, 2)
            x = torch.cat((x, lc_features), dim=1)

        x = self.layers(x)
        alphas = x[:, 0]
        betas = x[:, 1]
        gammas = x[:, 2]

        output = torch.add(
            alphas,
            torch.mul(
                self.last_act_func(torch.cat((betas, betas))),
                torch.pow(
                    predict_budgets,
                    torch.mul(self.last_act_func(torch.cat((gammas, gammas))), -1)
                )
            ),
        )

        return output

    def set_optimizer(self, hp):
        optimizer_class = self.get_class(torch.optim, hp.optimizer)
        self.optimizer = optimizer_class(self.parameters(), lr=hp.learning_rate)

    def get_class(self, package, name):
        return getattr(package, name)

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
        outputs = self((batch_examples, batch_budgets, batch_budgets, batch_curves))
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

        if reset_optimizer:
            self.set_optimizer(self.hp)

        self.set_seed(self.seed)
        patience_rounds = 0
        best_loss = np.inf
        best_state = deepcopy(self.state_dict())

        for epoch in range(0, nr_epochs):
            normalized_loss = self.train_epoch()
            self.logger.info(f'Epoch {epoch + 1}, Loss:{normalized_loss}')

            if self.hp.activate_early_stopping:
                if normalized_loss < best_loss:
                    best_state = deepcopy(self.state_dict())
                    best_loss = normalized_loss
                    patience_rounds = 0
                elif normalized_loss > best_loss:
                    patience_rounds += 1
                    if patience_rounds == self.hp.early_stopping_it:
                        self.load_state_dict(best_state)
                        self.logger.info(f'Stopping training since validation loss is not improving')
                        break
        check_seed_torch = torch.random.get_rng_state().sum()
        self.logger.info(f"end rng_state {check_seed_torch}")
        if self.hp.activate_early_stopping:
            self.load_state_dict(best_state)

    def __del__(self):
        ConditionedPowerLawModel._instance_counter = 0

    def print_parameters(self):
        for name, param in self.named_parameters():
            a = param
            print(f"{name}: {param}")
