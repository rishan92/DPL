import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
import wandb
import math

from src.models.power_law.power_law_model import PowerLawModel
from .scaling_layer import ScalingLayer


class TargetSpaceComplexPowerLawModel(PowerLawModel):
    param_names = ('alphas_r', 'y1', 'y2')

    @staticmethod
    def get_default_meta():
        hp = {
            'nr_units': 128,
            'nr_layers': 2,
            'kernel_size': 3,
            'nr_filters': 4,
            'nr_cnn_layers': 2,
            'use_learning_curve': False,
            'use_learning_curve_mask': False,
            'learning_rate': 0.001,
            'act_func': 'LeakyReLU',
            'last_act_func': 'Sigmoid',
            'output_act_func': None,
            'loss_function': 'L1Loss',
            'optimizer': 'Adam',
            'activate_early_stopping': False,
            'early_stopping_it': 0,
            'use_scaling_layer': False,
            'scaling_layer_bias_values': [0, 0, math.log(0.01) / math.log(1 / 51)]  # [0, 0, 1.17125493757]
        }
        return hp

    def get_linear_net(self):
        layers = []
        # adding one since we concatenate the features with the budget
        nr_initial_features = self.nr_features
        if self.meta.use_learning_curve:
            nr_initial_features = self.nr_features + self.meta.nr_filters

        layers.append(nn.Linear(nr_initial_features, self.meta.nr_units))
        layers.append(self.act_func)

        for i in range(2, self.meta.nr_layers + 1):
            layers.append(nn.Linear(self.meta.nr_units, self.meta.nr_units))
            layers.append(self.act_func)

        last_layer = nn.Linear(self.meta.nr_units, 3)
        layers.append(last_layer)

        if hasattr(self.meta, "use_scaling_layer") and self.meta.use_scaling_layer:
            bias_values = None
            if hasattr(self.meta, "scaling_layer_bias_values") and self.meta.scaling_layer_bias_values:
                bias_values = self.meta.scaling_layer_bias_values
            scaling_layer = ScalingLayer(in_features=3, bias_values=bias_values)
            layers.append(scaling_layer)

        net = torch.nn.Sequential(*layers)
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
        x, predict_budgets, learning_curves = batch

        x = self.linear_net(x)
        alphas = x[:, 0]
        y1 = x[:, 1]
        y2 = x[:, 2]

        y1 = self.last_act_func(y1)
        y2 = self.last_act_func(y2)

        # y2 = y1 * y2_diff

        val = (y2 - alphas) / (y1 - alphas)

        abs_val = torch.abs(val)
        log_abs_val = torch.log(abs_val)

        # Calculate the angle (imaginary part)
        angle = torch.atan2(torch.tensor(0.0), val)

        log_val = torch.complex(log_abs_val, angle)

        gammas = log_val / torch.log(torch.tensor(1 / 51))

        betas = y2 - alphas

        output_complex = torch.add(
            alphas,
            torch.mul(
                betas,
                torch.pow(
                    predict_budgets,
                    torch.mul(gammas, -1)
                )
            ),
        )
        output = output_complex.real
        if self.output_act_func and self.training:
            output = self.output_act_func(output)

        info = {
            'alpha': alphas,
            'beta': betas,
            'gamma': gammas,
            'pl_output': output,
        }

        return output, info
