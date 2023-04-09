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


class FeatureExtractor(nn.Module):
    """
    The feature extractor that is part of the deep kernel.
    """

    def __init__(self, configuration):
        super(FeatureExtractor, self).__init__()

        self.configuration = configuration

        self.nr_layers = configuration['nr_layers']
        self.act_func = nn.LeakyReLU()
        # adding one to the dimensionality of the initial input features
        # for the concatenation with the budget.
        initial_features = configuration['nr_initial_features'] + 1
        self.fc1 = nn.Linear(initial_features, configuration['layer1_units'])
        self.bn1 = nn.BatchNorm1d(configuration['layer1_units'])
        for i in range(2, self.nr_layers):
            setattr(
                self,
                f'fc{i + 1}',
                nn.Linear(configuration[f'layer{i - 1}_units'], configuration[f'layer{i}_units']),
            )
            setattr(
                self,
                f'bn{i + 1}',
                nn.BatchNorm1d(configuration[f'layer{i}_units']),
            )

        setattr(
            self,
            f'fc{self.nr_layers}',
            nn.Linear(
                configuration[f'layer{self.nr_layers - 1}_units'] +
                configuration['cnn_nr_channels'],  # accounting for the learning curve features
                configuration[f'layer{self.nr_layers}_units']
            ),
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, kernel_size=(configuration['cnn_kernel_size'],), out_channels=4),
            nn.AdaptiveMaxPool1d(1),
        )

    def forward(self, x, budgets, learning_curves):

        # add an extra dimensionality for the budget
        # making it nr_rows x 1.
        budgets = torch.unsqueeze(budgets, dim=1)
        # concatenate budgets with examples
        x = cat((x, budgets), dim=1)
        x = self.fc1(x)
        x = self.act_func(self.bn1(x))

        for i in range(2, self.nr_layers):
            x = self.act_func(
                getattr(self, f'bn{i}')(
                    getattr(self, f'fc{i}')(
                        x
                    )
                )
            )

        # add an extra dimensionality for the learning curve
        # making it nr_rows x 1 x lc_values.
        learning_curves = torch.unsqueeze(learning_curves, 1)
        lc_features = self.cnn(learning_curves)
        # revert the output from the cnn into nr_rows x nr_kernels.
        lc_features = torch.squeeze(lc_features, 2)

        # put learning curve features into the last layer along with the higher level features.
        x = cat((x, lc_features), dim=1)
        x = self.act_func(getattr(self, f'fc{self.nr_layers}')(x))

        return x
