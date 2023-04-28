import random
from copy import deepcopy
import logging
import os
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch import cat
from loguru import logger

import gpytorch
from src.models.base.base_pytorch_module import BasePytorchModule
import src.models.activation_functions
from src.utils.utils import get_class_from_package, get_class_from_packages
from abc import ABC, abstractmethod


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

        # set nr_units as a List
        self.meta.nr_units = self.get_layer_units()

        self.nr_initial_features = None

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
