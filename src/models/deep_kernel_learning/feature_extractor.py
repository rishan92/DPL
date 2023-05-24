import random
from copy import deepcopy
import logging
import os
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch import cat

import gpytorch
from src.models.base.base_pytorch_module import BasePytorchModule
import src.models.activation_functions
from src.utils.utils import get_class_from_package, get_class_from_packages
from src.models.deep_kernel_learning.base_feature_extractor import BaseFeatureExtractor


class FeatureExtractor(BaseFeatureExtractor):
    """
    The feature extractor that is part of the deep kernel.
    """

    def __init__(self, nr_features, seed=None):
        # adding one since we concatenate the features with the budget
        nr_initial_features = nr_features + 1
        super().__init__(nr_initial_features, seed=seed)

        # self.linear_net = self.get_linear_net()
        # self.after_cnn_linear_net = self.get_after_cnn_linear_net()
        #
        # if self.meta.use_learning_curve:
        #     self.cnn_net = self.get_cnn_net()
        self.set_seed(seed=seed)
        self.fc1 = nn.Linear(self.nr_initial_features, self.meta.nr_units[0])
        self.bn1 = nn.BatchNorm1d(self.meta.nr_units[0])
        for i in range(1, self.meta.nr_layers - 1):
            setattr(
                self,
                f'fc{i + 1}',
                nn.Linear(self.meta.nr_units[i - 1], self.meta.nr_units[i]),
            )
            setattr(
                self,
                f'bn{i + 1}',
                nn.BatchNorm1d(self.meta.nr_units[i]),
            )

        setattr(
            self,
            f'fc{self.meta.nr_layers}',
            nn.Linear(
                self.meta.nr_units[-2] +
                self.meta.cnn_nr_channels,  # accounting for the learning curve features
                self.meta.nr_units[-1]
            ),
        )
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, kernel_size=(self.meta.cnn_kernel_size,), out_channels=self.meta.cnn_nr_channels),
            nn.AdaptiveMaxPool1d(1),
        )

    @staticmethod
    def get_default_meta():
        hp = {
            'nr_layers': 2,
            'nr_units': [64, 128],
            'cnn_nr_channels': 4,
            'cnn_kernel_size': 3,
            'cnn_nr_layers': 1,
            'use_learning_curve': True,
            'use_learning_curve_mask': False,
            'act_func': 'LeakyReLU',
        }
        return hp

    def get_linear_net(self):
        layers = []

        layers.append(nn.Linear(self.nr_initial_features, self.meta.nr_units[0]))
        layers.append(nn.BatchNorm1d(self.meta.nr_units[0]))
        layers.append(self.act_func)

        for i in range(1, self.meta.nr_layers - 1):
            layers.append(nn.Linear(self.meta.nr_units[i - 1], self.meta.nr_units[i]))
            layers.append(nn.BatchNorm1d(self.meta.nr_units[i]))
            layers.append(self.act_func)

        net = torch.nn.Sequential(*layers)
        return net

    def get_after_cnn_linear_net(self):
        layers = []
        nr_features = self.meta.nr_units[-2]
        if self.meta.use_learning_curve:
            nr_features = self.meta.nr_units[-2] + self.meta.cnn_nr_channels

        layers.append(nn.Linear(nr_features, self.meta.nr_units[-1]))
        layers.append(self.act_func)

        net = torch.nn.Sequential(*layers)
        return net

    def get_cnn_net(self):
        cnn_part = []

        cnn_part.append(
            nn.Conv1d(
                in_channels=1,
                kernel_size=(self.meta.cnn_kernel_size,),
                out_channels=self.meta.cnn_nr_channels,
            ),
        )

        for i in range(1, self.meta.cnn_nr_layers):
            cnn_part.append(self.act_func)
            cnn_part.append(
                nn.Conv1d(
                    in_channels=self.meta.cnn_nr_channels,
                    kernel_size=(self.meta.cnn_kernel_size,),
                    out_channels=self.meta.cnn_nr_channels,
                ),
            ),

        cnn_part.append(nn.AdaptiveMaxPool1d(1))

        net = torch.nn.Sequential(*cnn_part)
        return net

    def forward(self, x, budgets, learning_curves):

        # add an extra dimensionality for the budget
        # making it nr_rows x 1.
        budgets = torch.unsqueeze(budgets, dim=1)
        # concatenate budgets with examples
        x = cat((x, budgets), dim=1)
        x = self.fc1(x)
        x = self.act_func(self.bn1(x))

        for i in range(1, self.meta.nr_layers - 1):
            x = self.act_func(
                getattr(self, f'bn{i + 1}')(
                    getattr(self, f'fc{i + 1}')(
                        x
                    )
                )
            )

        # add an extra dimensionality for the learning curve
        # making it nr_rows x 1 x lc_values.
        learning_curves = torch.unsqueeze(learning_curves, dim=1)
        lc_features = self.cnn(learning_curves)
        # revert the output from the cnn into nr_rows x nr_kernels.
        lc_features = torch.squeeze(lc_features, dim=2)

        # put learning curve features into the last layer along with the higher level features.
        x = cat((x, lc_features), dim=1)
        x = self.act_func(getattr(self, f'fc{self.meta.nr_layers}')(x))

        return x, None

    # def forward(self, x, budgets, learning_curves):
    #
    #     # add an extra dimensionality for the budget
    #     # making it nr_rows x 1.
    #     budgets = torch.unsqueeze(budgets, dim=1)
    #     # concatenate budgets with examples
    #     x = cat((x, budgets), dim=1)
    #     x = self.linear_net(x)
    #
    #     # add an extra dimensionality for the learning curve
    #     # making it nr_rows x 1 x lc_values.
    #     learning_curves = torch.unsqueeze(learning_curves, dim=1)
    #     lc_features = self.cnn_net(learning_curves)
    #     # revert the output from the cnn into nr_rows x nr_kernels.
    #     lc_features = torch.squeeze(lc_features, dim=2)
    #
    #     # put learning curve features into the last layer along with the higher level features.
    #     x = cat((x, lc_features), dim=1)
    #     x = self.after_cnn_linear_net(x)
    #
    #     return x, None
