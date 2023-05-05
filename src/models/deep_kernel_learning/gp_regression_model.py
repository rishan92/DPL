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


class GPRegressionModel(gpytorch.models.ExactGP):
    """
    A simple GP model.
    """

    def __init__(
        self,
        input_size: int,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        use_seperate_lengthscales: bool = False,
        use_scale_to_bounds: bool = False
    ):
        """
        Constructor of the GPRegressionModel.

        Args:
            train_x: The initial train examples for the GP.
            train_y: The initial train labels for the GP.
            likelihood: The likelihood to be used.
        """
        train_x = torch.ones(input_size, input_size)
        train_y = torch.ones(input_size)

        super().__init__(train_x, train_y, likelihood)

        self.use_separate_lengthscales = use_seperate_lengthscales
        self.use_scale_to_bounds = use_scale_to_bounds

        self.mean_module = gpytorch.means.ConstantMean()
        if use_seperate_lengthscales:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=input_size))
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        if self.use_scale_to_bounds:
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        if self.use_scale_to_bounds:
            x = self.scale_to_bounds(x)

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
