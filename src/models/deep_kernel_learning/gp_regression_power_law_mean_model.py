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
from .power_law_mean import PowerLawMean


class GPRegressionPowerLawMeanModel(gpytorch.models.ExactGP):
    """
    A simple GP model.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        seperate_lengthscales: bool = False
    ):
        """
        Constructor of the GPRegressionModel.

        Args:
            train_x: The initial train examples for the GP.
            train_y: The initial train labels for the GP.
            likelihood: The likelihood to be used.
        """
        super().__init__(train_x, train_y, likelihood)

        self.seperate_lengthscales = seperate_lengthscales

        self.mean_module = PowerLawMean()
        if seperate_lengthscales:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=4))
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        # grid_size = gpytorch.utils.grid.choose_grid_size(train_x, 1.0)
        #
        # self.covar_module = gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.GridInterpolationKernel(
        #         gpytorch.kernels.RBFKernel(ard_num_dims=3), grid_size=grid_size, num_dims=3
        #     )
        # )

    def forward(self, x):
        mean_x = self.mean_module(x)
        # x_wo_output = x[:, :-1]
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
