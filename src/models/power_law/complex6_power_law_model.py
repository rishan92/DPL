import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
import wandb
import math

from src.models.power_law.power_law_model import PowerLawModel
from .scaling_layer import ScalingLayer
from torch.autograd import grad


class Complex6PowerLawModel(PowerLawModel):
    param_names = ('alphas_r', 'betas_r', 'gammas_r', 'alphas_i', 'betas_i', 'gammas_i')

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
            'last_act_func': 'SelfGLU',
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

        last_layer = nn.Linear(self.meta.nr_units, 6)
        layers.append(last_layer)

        if hasattr(self.meta, "use_scaling_layer") and self.meta.use_scaling_layer:
            bias_values = None
            if hasattr(self.meta, "scaling_layer_bias_values") and self.meta.scaling_layer_bias_values:
                bias_values = self.meta.scaling_layer_bias_values
            scaling_layer = ScalingLayer(in_features=6, bias_values=bias_values)
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

        # x = torch.cat((x, torch.unsqueeze(evaluated_budgets, 1)), dim=1)
        if self.meta.use_learning_curve:
            lc_features = self.cnn_net(learning_curves)
            # revert the output from the cnn into nr_rows x nr_kernels.
            lc_features = torch.squeeze(lc_features, 2)
            x = torch.cat((x, lc_features), dim=1)

        x = self.linear_net(x)
        alphas_r_b = x[:, 0]
        betas_r_b = x[:, 1]
        gammas_r_b = x[:, 2]
        alphas_i_b = x[:, 3]
        betas_i_b = x[:, 4]
        gammas_i_b = x[:, 5]

        alphas_r = alphas_r_b
        alphas_i = alphas_i_b
        betas_r = self.last_act_func(betas_r_b)
        betas_i = self.last_act_func(betas_i_b)
        gammas_r = self.last_act_func(gammas_r_b)
        gammas_i = self.last_act_func(gammas_i_b)

        alphas = alphas_r + 1j * alphas_i
        betas = betas_r + 1j * betas_i
        gammas = gammas_r + 1j * gammas_i

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

        # do_dar, = grad(output, alphas_r_b, create_graph=True)
        # do_dbr, = grad(output, betas_r_b, create_graph=True)
        # do_dgr, = grad(output, gammas_r_b, create_graph=True)
        # do_dai, = grad(output, alphas_i_b, create_graph=True)
        # do_dbi, = grad(output, betas_i_b, create_graph=True)
        # do_dgi, = grad(output, gammas_i_b, create_graph=True)
        # print(f"{do_dar=}")
        # print(f"{do_dbr=}")
        # print(f"{do_dgr=}")
        # print(f"{do_dai=}")
        # print(f"{do_dbi=}")
        # print(f"{do_dgi=}")

        info = {
            'alpha': alphas,
            'beta': betas,
            'gamma': gammas,
            'pl_output': output,
        }

        return output, info
