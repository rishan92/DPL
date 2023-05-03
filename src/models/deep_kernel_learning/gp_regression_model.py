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
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
