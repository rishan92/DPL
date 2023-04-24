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

import gpytorch
from src.models.activation_functions import SelfGLU, Abs, SelfAbsGLU


class FeatureExtractorPowerLaw(nn.Module):
    """
    The feature extractor that is part of the deep kernel.
    """

    def __init__(self, configuration):
        super().__init__()

        self.meta = configuration

        self.nr_layers = self.meta.nr_layers
        self.act_func = nn.LeakyReLU()
        self.last_act_func = SelfGLU()
        # adding one to the dimensionality of the initial input features
        # for the concatenation with the budget.
        self.nr_features = self.meta.nr_features

        self.layers = self.get_linear_net()

    def get_linear_net(self):
        layers = []
        # adding one since we concatenate the features with the budget
        nr_initial_features = self.nr_features

        layers.append(nn.Linear(nr_initial_features, self.meta.nr_units))
        layers.append(self.act_func)

        for i in range(2, self.meta.nr_layers + 1):
            layers.append(nn.Linear(self.meta.nr_units, self.meta.nr_units))
            layers.append(self.act_func)

        last_layer = nn.Linear(self.meta.nr_units, 3)
        layers.append(last_layer)

        net = torch.nn.Sequential(*layers)
        return net

    def forward(self, x, budgets, learning_curves):
        x = self.layers(x)

        alphas = x[:, 0]
        betas = x[:, 1]
        gammas = x[:, 2]
        betas = self.last_act_func(betas)
        gammas = self.last_act_func(gammas)

        output = torch.add(
            alphas,
            torch.mul(
                betas,  # torch.mul(betas, -1),
                torch.pow(
                    budgets,
                    torch.mul(gammas, -1)
                )
            ),
        )
        budgets = torch.unsqueeze(budgets, dim=1)
        alphas = torch.unsqueeze(alphas, dim=1)
        betas = torch.unsqueeze(betas, dim=1)
        gammas = torch.unsqueeze(gammas, dim=1)
        output = torch.unsqueeze(output, dim=1)

        x = cat((alphas, betas, gammas, output), dim=1)
        # x = cat((budgets, output), dim=1)
        # x = output

        # budget_lower_limit = torch.tensor(1 / 51)
        # budget_upper_limit = torch.tensor(1)

        # constrained_alpha = alphas
        # constrained_beta = betas
        # constrained_gamma = gammas
        # start_output = torch.add(
        #     constrained_alpha,
        #     torch.mul(
        #         constrained_beta,
        #         torch.pow(
        #             budget_lower_limit,
        #             torch.mul(constrained_gamma, -1)
        #         )
        #     ),
        # )
        # end_output = torch.add(
        #     constrained_alpha,
        #     torch.mul(
        #         constrained_beta,
        #         torch.pow(
        #             budget_upper_limit,
        #             torch.mul(constrained_gamma, -1)
        #         )
        #     ),
        # )
        # print(f"start {start_output}")
        # print(f"end {end_output}")

        return x
