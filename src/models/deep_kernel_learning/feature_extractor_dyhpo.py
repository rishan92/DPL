import random
from copy import deepcopy
import logging
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import cat
from loguru import logger
import math

import gpytorch
from src.models.deep_kernel_learning.base_feature_extractor import BaseFeatureExtractor
from src.models.layers.scaling_layer import ScalingLayer


class FeatureExtractorDYHPO(BaseFeatureExtractor):
    """
    The feature extractor that is part of the deep kernel.
    """
    param_names = ('alphas_r', 'betas_r', 'gammas_r')

    def __init__(self, nr_features, seed=None):
        super().__init__(nr_features, seed=seed)

        # self.linear_net = self.get_linear_net()

        # self.fc1 = nn.Linear(self.nr_initial_features, self.meta.nr_units[0])
        # self.bn1 = nn.BatchNorm1d(self.meta.nr_units[0])
        # for i in range(1, self.meta.nr_layers - 1):
        #     setattr(
        #         self,
        #         f'fc{i + 1}',
        #         nn.Linear(self.meta.nr_units[i - 1], self.meta.nr_units[i]),
        #     )
        #     setattr(
        #         self,
        #         f'bn{i + 1}',
        #         nn.BatchNorm1d(self.meta.nr_units[i]),
        #     )
        #
        # setattr(
        #     self,
        #     f'fc{self.meta.nr_layers}',
        #     nn.Linear(
        #         self.meta.nr_units[-2],
        #         # self.meta.cnn_nr_channels,  # accounting for the learning curve features
        #         self.meta.nr_units[-1]
        #     ),
        # )
        #
        # setattr(
        #     self,
        #     f'fc{self.meta.nr_layers + 1}',
        #     nn.Linear(self.meta.nr_units[-1], 3),
        # )

    @staticmethod
    def get_default_meta():
        hp = {
            'nr_layers': 2,
            'nr_units': [64, 128],
            'cnn_nr_channels': 4,
            'cnn_kernel_size': 3,
            'cnn_nr_layers': 1,
            'dropout_rate': 0,
            'use_batch_norm': True,
            'use_learning_curve': False,
            'use_learning_curve_mask': False,
            'act_func': 'LeakyReLU',
            'last_act_func': 'Identity',
            'alpha_act_func': 'Identity',
            'beta_act_func': 'GELU',
            'gamma_act_func': 'GELU',
            'output_act_func': None,
            'alpha_beta_is_difference': False,
            'use_scaling_layer': False,
            'scaling_layer_bias_values': [0, 0, math.log(0.01) / math.log(1 / 51)]  # [0, 0, 1.17125493757],
        }
        return hp

    # def forward(self, x, budgets, learning_curves):
    #     x = self.fc1(x)
    #     x = self.act_func(self.bn1(x))
    #
    #     for i in range(1, self.meta.nr_layers - 1):
    #         x = self.act_func(
    #             getattr(self, f'bn{i + 1}')(
    #                 getattr(self, f'fc{i + 1}')(
    #                     x
    #                 )
    #             )
    #         )
    #
    #     # # add an extra dimensionality for the learning curve
    #     # # making it nr_rows x 1 x lc_values.
    #     # learning_curves = torch.unsqueeze(learning_curves, dim=1)
    #     # lc_features = self.cnn(learning_curves)
    #     # # revert the output from the cnn into nr_rows x nr_kernels.
    #     # lc_features = torch.squeeze(lc_features, dim=2)
    #
    #     # # put learning curve features into the last layer along with the higher level features.
    #     # x = cat((x, lc_features), dim=1)
    #
    #     x = self.act_func(getattr(self, f'fc{self.meta.nr_layers}')(x))
    #
    #     x = getattr(self, f'fc{self.meta.nr_layers + 1}')(x)
    #
    #     alphas = x[:, 0]
    #     betas = x[:, 1]
    #     gammas = x[:, 2]
    #     betas = self.last_act_func(betas)
    #     gammas = self.last_act_func(gammas)
    #
    #     output = torch.add(
    #         alphas,
    #         torch.mul(
    #             betas,
    #             torch.pow(
    #                 budgets,
    #                 torch.mul(gammas, -1)
    #             )
    #         ),
    #     )
    #
    #     info = {
    #         'alpha': alphas,
    #         'beta': betas,
    #         'gamma': gammas,
    #         'pl_output': output,
    #     }
    #
    #     budgets = torch.unsqueeze(budgets, dim=1)
    #     alphas = torch.unsqueeze(alphas, dim=1)
    #     betas = torch.unsqueeze(betas, dim=1)
    #     gammas = torch.unsqueeze(gammas, dim=1)
    #     output = torch.unsqueeze(output, dim=1)
    #
    #     x = cat((alphas, betas, gammas, output), dim=1)
    #
    #     return x, info

    def get_linear_net(self):
        layers = []

        layers.append(nn.Linear(self.nr_initial_features, self.meta.nr_units[0]))
        if hasattr(self.meta, 'use_batch_norm') and self.meta.use_batch_norm:
            layers.append(nn.BatchNorm1d(self.meta.nr_units[0]))
        layers.append(self.act_func)
        if hasattr(self.meta, 'dropout_rate') and self.meta.dropout_rate != 0:
            layers.append(nn.Dropout(self.meta.dropout_rate))

        for i in range(1, self.meta.nr_layers):
            layers.append(nn.Linear(self.meta.nr_units[i - 1], self.meta.nr_units[i]))
            if hasattr(self.meta, 'use_batch_norm') and self.meta.use_batch_norm:
                layers.append(nn.BatchNorm1d(self.meta.nr_units[i]))
            layers.append(self.act_func)
            if hasattr(self.meta, 'dropout_rate') and self.meta.dropout_rate != 0:
                layers.append(nn.Dropout(self.meta.dropout_rate))

        layers.append(nn.Linear(self.meta.nr_units[-1], 3))

        if hasattr(self.meta, "use_scaling_layer") and self.meta.use_scaling_layer:
            bias_values = None
            if hasattr(self.meta, "scaling_layer_bias_values") and self.meta.scaling_layer_bias_values:
                bias_values = self.meta.scaling_layer_bias_values
            scaling_layer = ScalingLayer(in_features=3, bias_values=bias_values)
            layers.append(scaling_layer)

        net = torch.nn.Sequential(*layers)
        return net

    def forward(self, x, budgets, learning_curves):
        x = self.linear_net(x)

        alphas = x[:, 0]
        betas = x[:, 1]
        gammas = x[:, 2]

        alphas = self.alpha_act_func(alphas)
        betas = self.beta_act_func(betas)
        gammas = self.gamma_act_func(gammas)

        if hasattr(self.meta, 'alpha_beta_is_difference') and self.meta.alpha_beta_is_difference:
            alphas_plus_beta = alphas
            alpha_weight = betas
            alphas = alphas_plus_beta * alpha_weight
            betas = alphas_plus_beta * (1 - alpha_weight)

        output = torch.add(
            alphas,
            torch.mul(
                betas,
                torch.pow(
                    budgets,
                    torch.mul(gammas, -1)
                )
            ),
        )
        if self.output_act_func:
            output = self.output_act_func(output)

        info = {
            'alpha': alphas,
            'beta': betas,
            'gamma': gammas,
            'pl_output': output,
        }

        alphas = torch.unsqueeze(alphas, dim=1)
        betas = torch.unsqueeze(betas, dim=1)
        gammas = torch.unsqueeze(gammas, dim=1)
        output = torch.unsqueeze(output, dim=1)

        x = cat((alphas, betas, gammas, output), dim=1)

        return x, info
