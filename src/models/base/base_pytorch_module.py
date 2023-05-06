from abc import ABC

import torch
import torch.nn as nn
import numpy as np
import random
from typing import Dict, Any, Optional
from loguru import logger
from src.data_loader.tabular_data_loader import WrappedDataLoader
from types import SimpleNamespace
from torch.utils.data import DataLoader
import wandb
from functools import partial

from src.data_loader.surrogate_data_loader import SurrogateDataLoader
from src.utils.utils import classproperty
from .meta import Meta
from abc import ABC, abstractmethod
import global_variables as gv


class BasePytorchModule(nn.Module, Meta, ABC):
    meta = None
    _view_gradient_step = 0

    def __init__(self, nr_features, seed=None, checkpoint_path='.'):
        super().__init__()

        self.train_dataset = None
        self.train_dataloader = None
        self.train_dataloader_it = None

        self.checkpoint_path = checkpoint_path

        self.logger = logger

        assert self.meta is not None, "Meta parameters are not set"

        self.nr_features = nr_features
        self.seed = seed
        self.set_seed(self.seed)

    @staticmethod
    def set_seed(seed):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def set_dataloader(self, train_dataloader):
        assert train_dataloader is not None
        self.train_dataloader = train_dataloader
        self.train_dataloader_it = iter(self.train_dataloader)

    @classproperty
    def meta_use_learning_curve(cls):
        if hasattr(cls.meta, 'use_learning_curve'):
            return cls.meta.use_learning_curve
        else:
            return False

    @classproperty
    def meta_use_learning_curve_mask(cls):
        if hasattr(cls.meta, 'use_learning_curve_mask'):
            return cls.meta.use_learning_curve_mask
        else:
            return False

    @classproperty
    def meta_output_act_func(cls):
        if hasattr(cls.meta, 'output_act_func'):
            return cls.meta.output_act_func
        else:
            return None

    @classproperty
    def meta_clip_gradients(cls):
        if hasattr(cls.meta, 'clip_gradients'):
            return cls.meta.clip_gradients
        else:
            return 0

    @classproperty
    def meta_cnn_kernel_size(cls):
        if hasattr(cls.meta, 'cnn_kernel_size'):
            return cls.meta.cnn_kernel_size
        else:
            return 0

    def print_parameters(self):
        for name, param in self.named_parameters():
            print(f"{name}: {param}")

    @staticmethod
    def gradient_logging_hook(module, grad_input, grad_output, names):
        BasePytorchModule._view_gradient_step += 1

        grads = grad_output[0]
        grads = torch.abs(grads).mean(dim=0)
        grads = grads.numpy()

        if gv.IS_WANDB and len(names) > 0:
            wandb_data = {
                'gradients_flow/gradient_step': BasePytorchModule._view_gradient_step
            }
            for i, name in enumerate(names):
                wandb_data[f'gradients_flow/{name}'] = grads[i]
            wandb.log(wandb_data)
