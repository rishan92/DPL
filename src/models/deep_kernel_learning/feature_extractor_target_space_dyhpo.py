import random
from copy import deepcopy
import logging
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import cat
import math

import gpytorch
from src.models.deep_kernel_learning.base_feature_extractor import BaseFeatureExtractor
from src.models.layers.scaling_layer import ScalingLayer


class FeatureExtractorTargetSpaceDYHPO(BaseFeatureExtractor):
    """
    The feature extractor that is part of the deep kernel.
    """
    param_names = ('alphas_r', 'betas_r', 'gammas_r')

    def __init__(self, nr_features, seed=None):
        super().__init__(nr_features, seed=seed)

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
            'nr_units': [128, 128],
            'cnn_nr_channels': 4,
            'cnn_kernel_size': 3,
            'cnn_nr_layers': 1,
            'dropout_rate': 0,
            'use_batch_norm': False,
            'use_learning_curve': False,
            'use_learning_curve_mask': False,
            'act_func': 'LeakyReLU',
            'last_act_func': 'Identity',
            'alpha_act_func': 'BoundedReLU',
            'beta_act_func': 'BoundedReLU',
            'gamma_act_func': 'BoundedReLU',
            'output_act_func': None,
            'alpha_beta_is_difference': None,  # null "half"  "full"
            'use_gamma_constraint': "flip2",  # null "positive"  "half"  "full" "full_flip" "flip"
            'use_gamma_positive': False,
            'use_complex': False,
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
            # if hasattr(self.meta, 'use_batch_norm') and self.meta.use_batch_norm:
            #     layers.append(nn.BatchNorm1d(self.meta.nr_units[i]))
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
        y1 = x[:, 1]
        y2 = x[:, 2]

        alphas = self.alpha_act_func(alphas)
        y1 = self.beta_act_func(y1)
        y2 = self.gamma_act_func(y2)

        if hasattr(self.meta, 'alpha_beta_is_difference') and self.meta.alpha_beta_is_difference is not None:
            if self.meta.alpha_beta_is_difference == 'half':
                y2 = y1 * y2
            elif self.meta.alpha_beta_is_difference == 'full':
                y2_prev = y2
                y2 = torch.where(y2 <= y1, y1 * y2, y2)
                y1 = torch.where(y2_prev > y1, y1 * y2_prev, y1)
            else:
                raise NotImplementedError

        if hasattr(self.meta, 'use_gamma_constraint'):
            if self.meta.use_gamma_constraint is None:
                lb = 0
                ub = 1
                alphas = alphas * (ub - lb) + lb
            elif self.meta.use_gamma_constraint == 'positive':
                lm = torch.min(y2, y1)
                alphas = alphas * lm
            elif self.meta.use_gamma_constraint == 'half':
                lm = torch.min(y2, y1)
                lb = -1
                ub = 1

                alphas = alphas * (ub - lb) + lb

                m = ub
                alphas = (alphas - lb) * (lm - lb) / (m - lb) + lb
            elif self.meta.use_gamma_constraint == "full":
                lm = torch.min(y2, y1)
                um = torch.max(y1, y2)
                lb = -1
                ub = 2

                alphas = alphas * (ub - lb) + lb

                m = (y2 + y1) / 2
                mask = alphas <= m

                lower_transform = (alphas - lb) * (lm - lb) / (m - lb) + lb
                upper_transform = (alphas - m) * (ub - um) / (ub - m) + um
                alphas = torch.where(mask, lower_transform, upper_transform)
            elif self.meta.use_gamma_constraint == "full_flip":
                lm = torch.min(y2, y1)
                um = torch.max(y1, y2)
                lb = -1
                ub = 2

                alphas = alphas * (ub - lb) + lb

                m = (y2 + y1) / 2
                mask = alphas <= m

                lower_transform = (alphas - lb) * (lm - lb) / (m - lb) + lb
                upper_transform = (alphas - m) * (ub - um) / (ub - m) + um
                alphas = torch.where(mask, lower_transform, upper_transform)

                # flip
                a_lower = (lb + lm) / 2
                a_upper = (um + ub) / 2
                alphas = torch.where(mask, 2 * a_lower - alphas, 2 * a_upper - alphas)
            elif self.meta.use_gamma_constraint == 'flip':
                alphas = torch.where(y2 <= y1, alphas * y2, y2 + alphas * (1 - y2))
            elif self.meta.use_gamma_constraint == 'flip2':
                alphas = torch.where(y2 <= y1, alphas * y2, y2 + (1 - alphas) * (1 - y2))
            else:
                raise NotImplementedError

        val = ((y2 - alphas) / (y1 - alphas + torch.tensor(1e-4))) + torch.tensor(1e-4)

        abs_val = torch.abs(val)
        log_abs_val = torch.log(abs_val)

        if self.meta.use_complex:
            # Calculate the angle (imaginary part)
            angle = torch.atan2(torch.tensor(0.0), val)
            log_abs_val = torch.complex(log_abs_val, angle)

        gammas = log_abs_val / torch.log(torch.tensor(1 / 51))

        if hasattr(self.meta, 'use_gamma_positive') and self.meta.use_gamma_positive:
            gammas = torch.abs(gammas)

        betas = y2 - alphas

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
        if self.meta.use_complex and not self.training:
            # output2 = torch.add(
            #     alphas.detach(),
            #     torch.mul(
            #         betas.detach(),
            #         torch.pow(
            #             predict_budgets,
            #             torch.mul(gammas.real.detach(), -1)
            #         )
            #     ),
            # )
            output = output.real

        if self.output_act_func:
            output = self.output_act_func(output)

        info = {
            'alpha': alphas,
            'beta': betas,
            'gamma': gammas,
            'pl_output': output,
            'y1': y1,
            'y2': y2,
        }

        alphas = torch.unsqueeze(alphas, dim=1)
        betas = torch.unsqueeze(betas, dim=1)
        gammas = torch.unsqueeze(gammas, dim=1)
        output = torch.unsqueeze(output, dim=1)

        x = cat((alphas, betas, gammas, output), dim=1)

        return x, info
