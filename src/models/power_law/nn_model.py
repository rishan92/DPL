import math
import torch
import torch.nn as nn
from torch import cat
from copy import deepcopy
import numpy as np
import wandb
from abc import ABC, abstractmethod
from functools import partial
from typing import List, Tuple, Dict, Optional, Any, Union, Type
import inspect
from torch.nn.utils import clip_grad_norm_
import warnings
from loguru import logger
import torch.nn.functional as F
from scipy.stats import spearmanr
import properscoring as ps

from src.models.power_law.power_law_model import PowerLawModel
from src.models.layers.scaling_layer import ScalingLayer
from src.models.base.base_pytorch_module import BasePytorchModule
import src.models.activation_functions
from src.utils.utils import get_class_from_package, get_class_from_packages, get_inverse_function_class, \
    weighted_spearman
import global_variables as gv
from src.utils.torch_lr_finder import LRFinder
from src.utils.utils import acq


class NNModel(PowerLawModel):
    def __init__(
        self,
        nr_features,
        max_instances,
        nr_fidelity=1,
        seed=None,
        checkpoint_path='.'
    ):
        super().__init__(nr_features=nr_features, max_instances=max_instances, seed=seed,
                         checkpoint_path=checkpoint_path, nr_fidelity=nr_fidelity)

    @staticmethod
    def get_default_meta():
        hp = {
            'nr_units': 128,
            'nr_layers': 2,
            'kernel_size': 3,
            'nr_filters': 4,
            'nr_cnn_layers': 2,
            'dropout_rate': 0,
            'use_mc_dropout': False,
            'num_mc_dropout': 1,
            'use_batch_norm': False,
            'use_learning_curve': False,
            'use_learning_curve_mask': False,
            'use_suggested_learning_rate': False,
            'use_sample_weights': False,
            'use_sample_weight_by_label': False,
            'use_sample_weight_by_budget': False,
            'sample_weight_by_budget_strategy': None,
            'weight_regularization_factor': 0,
            'alpha_beta_constraint_factor': 0,
            'gamma_constraint_factor': 0,
            'output_constraint_factor': 0,
            'learning_rate': 1e-3,
            'refine_learning_rate': 1e-3,
            'act_func': 'LeakyReLU',
            'last_act_func': 'Identity',
            'alpha_act_func': 'Identity',
            'beta_act_func': 'Identity',
            'gamma_act_func': 'Identity',
            'output_act_func': None,
            'alpha_beta_is_difference': False,
            'use_gamma_constraint': False,
            'loss_function': 'L1Loss',
            'optimizer': 'Adam',
            'learning_rate_scheduler': None,
            # 'CosineAnnealingLR' 'LambdaLR' 'OneCycleLR' 'ExponentialLR'
            'learning_rate_scheduler_args': {
                'total_iters_factor': 1.0,
                'eta_min': 1e-6,
                'max_lr': 1e-4,
                'refine_max_lr': 1e-3,
                'exp_min': 1e-5,
                'refine_exp_min': 1e-6,
            },
            'activate_early_stopping': False,
            'early_stopping_it': 0,
            'use_scaling_layer': False,
            'scaling_layer_bias_values': [0, 0, math.log(0.01) / math.log(1 / 51)]  # [0, 0, 1.17125493757],
        }
        return hp

    def get_linear_net(self):
        layers = []
        # adding one since we concatenate the features with the budget
        nr_initial_features = self.nr_features + self.nr_fidelity
        if self.meta.use_learning_curve:
            nr_initial_features = self.nr_features + self.meta.nr_filters

        layers.append(nn.Linear(nr_initial_features, self.meta.nr_units))
        if hasattr(self.meta, 'use_batch_norm') and self.meta.use_batch_norm:
            layers.append(nn.BatchNorm1d(self.meta.nr_units))
        layers.append(self.act_func)
        if hasattr(self.meta, 'dropout_rate') and self.meta.dropout_rate != 0:
            layers.append(nn.Dropout(self.meta.dropout_rate))

        for i in range(2, self.meta.nr_layers + 1):
            layers.append(nn.Linear(self.meta.nr_units, self.meta.nr_units))
            if hasattr(self.meta, 'use_batch_norm') and self.meta.use_batch_norm:
                layers.append(nn.BatchNorm1d(self.meta.nr_units))
            layers.append(self.act_func)
            if hasattr(self.meta, 'dropout_rate') and self.meta.dropout_rate != 0:
                layers.append(nn.Dropout(self.meta.dropout_rate))

        output_size = 1
        last_layer = nn.Linear(self.meta.nr_units, output_size)
        layers.append(last_layer)

        if hasattr(self.meta, "use_scaling_layer") and self.meta.use_scaling_layer:
            bias_values = None
            if hasattr(self.meta, "scaling_layer_bias_values") and self.meta.scaling_layer_bias_values:
                bias_values = self.meta.scaling_layer_bias_values
            scaling_layer = ScalingLayer(in_features=output_size, bias_values=bias_values)
            layers.append(scaling_layer)

        net = torch.nn.Sequential(*layers)
        return net

    def get_cnn_net(self):
        cnn_part = []

        cnn_part.append(
            nn.Conv1d(
                in_channels=2,
                kernel_size=(self.meta.kernel_size,),
                out_channels=self.meta.nr_filters,
            ),
        )
        for i in range(1, self.meta.nr_cnn_layers):
            cnn_part.append(self.act_func)
            cnn_part.append(
                nn.Conv1d(
                    in_channels=self.meta.nr_filters,
                    kernel_size=(self.meta.kernel_size,),
                    out_channels=self.meta.nr_filters,
                ),
            ),
        cnn_part.append(nn.AdaptiveAvgPool1d(1))

        net = torch.nn.Sequential(*cnn_part)
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
        x, budgets, learning_curves = batch

        # x = torch.cat((x, torch.unsqueeze(evaluated_budgets, 1)), dim=1)
        # if self.meta.use_learning_curve:
        #     lc_features = self.cnn_net(learning_curves)
        #     # revert the output from the cnn into nr_rows x nr_kernels.
        #     lc_features = torch.squeeze(lc_features, 2)
        #     x = torch.cat((x, lc_features), dim=1)

        # budgets = torch.unsqueeze(predict_budgets, dim=1)
        # concatenate budgets with examples
        x = cat((x, budgets), dim=1)
        output = self.linear_net(x)
        output = output[:, 0]

        if self.alpha_act_func:
            output = self.alpha_act_func(output)

        if self.output_act_func and self.training:
            output = self.output_act_func(output)

        info = {
            'pl_output': output,
        }

        return output, info
