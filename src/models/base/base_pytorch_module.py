import torch
import torch.nn as nn
import numpy as np
import random
from typing import Dict, Any, Optional
from loguru import logger
from src.data_loader.tabular_data_loader import WrappedDataLoader
from types import SimpleNamespace
from torch.utils.data import DataLoader
from src.data_loader.surrogate_data_loader import SurrogateDataLoader
from src.utils.utils import classproperty


class BasePytorchModule(nn.Module):
    meta = None

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

    def get_meta(self):
        return vars(self.meta)

    @staticmethod
    def get_default_meta() -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def set_meta(cls, config=None):
        config = {} if config is None else config
        default_meta = cls.get_default_meta()
        meta = {**default_meta, **config}
        cls.meta = SimpleNamespace(**meta)
        return meta

    def set_seed(self, seed):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def set_dataloader(self, train_dataloader):
        assert train_dataloader is not None
        self.train_dataloader = train_dataloader
        self.train_dataloader_it = iter(self.train_dataloader)

    @classproperty
    def use_learning_curve(cls):
        return cls.meta.use_learning_curve

    @classproperty
    def use_learning_curve_mask(cls):
        return cls.meta.use_learning_curve_mask

    def print_parameters(self):
        for name, param in self.named_parameters():
            print(f"{name}: {param}")
