import random
from copy import deepcopy
import logging
import os
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import warnings
from functools import partial
from abc import ABC, abstractmethod
import gpytorch

from src.models.base.base_pytorch_module import BasePytorchModule
import src.models.activation_functions
from src.utils.utils import get_class_from_package, get_class_from_packages, get_inverse_function_class
import global_variables as gv
from src.utils.torch_lr_finder import LRFinder


class BaseFeatureExtractor(BasePytorchModule, ABC):
    """
    The feature extractor that is part of the deep kernel.
    """

    def __init__(self, nr_features, seed=None):
        super().__init__(nr_features, seed=seed)

        self.act_func = None
        self.last_act_func = None
        self.alpha_act_func = None
        self.beta_act_func = None
        self.gamma_act_func = None
        self.linear_net = None
        self.after_cnn_linear_net = None
        self.cnn_net = None
        self.output_act_func = None
        self.output_act_inverse_func = None

        if hasattr(self.meta, "act_func"):
            self.act_func = get_class_from_packages([torch.nn, src.models.activation_functions], self.meta.act_func)()
        if hasattr(self.meta, "last_act_func"):
            self.last_act_func = get_class_from_packages([torch.nn, src.models.activation_functions],
                                                         self.meta.last_act_func)()
        if hasattr(self.meta, "alpha_act_func"):
            self.alpha_act_func = get_class_from_packages([torch.nn, src.models.activation_functions],
                                                          self.meta.alpha_act_func)()
        if hasattr(self.meta, "beta_act_func"):
            self.beta_act_func = get_class_from_packages([torch.nn, src.models.activation_functions],
                                                         self.meta.beta_act_func)()
        if hasattr(self.meta, "gamma_act_func"):
            self.gamma_act_func = get_class_from_packages([torch.nn, src.models.activation_functions],
                                                          self.meta.gamma_act_func)()

        if hasattr(self.meta, "output_act_func") and self.meta.output_act_func:
            self.output_act_func = get_class_from_packages([torch.nn, src.models.activation_functions],
                                                           self.meta.output_act_func)()

            output_act_inverse_class = get_inverse_function_class(self.meta.output_act_func)
            self.output_act_inverse_func = output_act_inverse_class() if output_act_inverse_class else None

        self.use_representation_units = 0
        if hasattr(self.meta, 'use_representation_units') and self.meta.use_representation_units:
            if self.meta.use_representation_units == True:
                self.use_representation_units = self.nr_features
            else:
                self.use_representation_units = self.meta.use_representation_units

        # set nr_units as a List
        self.meta.nr_units = self.get_layer_units()

        self.nr_initial_features = nr_features

        self.linear_net = self.get_linear_net()

        if callable(getattr(self, "get_after_cnn_linear_net", None)):
            self.after_cnn_linear_net = self.get_after_cnn_linear_net()

        if hasattr(self.meta, "use_learning_curve") and self.meta.use_learning_curve:
            self.cnn_net = self.get_cnn_net()

        self.has_batchnorm_layers = False

        self.hook_handle = None

        self.post_init()

    def post_init(self):
        self.has_batchnorm_layers = self.get_has_batchnorm_layers()
        self.has_batchnorm_layers = True

        if gv.PLOT_GRADIENTS:
            if hasattr(self, 'param_names'):
                hook = partial(self.gradient_logging_hook, names=self.param_names)
                if self.after_cnn_linear_net is not None:
                    self.hook_handle = self.after_cnn_linear_net.register_full_backward_hook(hook=hook)
                if self.linear_net is not None:
                    self.hook_handle = self.linear_net.register_full_backward_hook(hook=hook)
                else:
                    warnings.warn("Gradient flow tracking with wandb is not supported for this module.")

    def get_has_batchnorm_layers(self):
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                return True
        return False

    def get_layer_units(self):
        if isinstance(self.meta.nr_units, List):
            nr_units = self.meta.nr_units
            assert len(nr_units) == self.meta.nr_layers, \
                f"given nr_units for {len(nr_units)} layers. Got {self.meta.nr_layers} layers"
        elif isinstance(self.meta.nr_units, int):
            nr_units = [self.meta.nr_units] * self.meta.nr_layers
        else:
            raise NotImplementedError

        return nr_units

    def hook_remove(self):
        if self.hook_handle:
            self.hook_handle.remove()
