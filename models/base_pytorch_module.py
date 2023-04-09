import torch
import torch.nn as nn
import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
from loguru import logger
from torch.utils.data import DataLoader
from data_loader.tabular_data_loader import WrappedDataLoader
from types import SimpleNamespace
from torch.utils.data import Dataset, DataLoader
from data_loader.surrogate_data_loader import SurrogateDataLoader
from copy import deepcopy


class BasePytorchModule(nn.Module):
    def __init__(self, nr_features, train_dataloader: SurrogateDataLoader = None, surrogate_configs=None):
        super().__init__()
        self.train_dataloader: Optional[DataLoader] = train_dataloader
        self.train_dataloader_it = iter(train_dataloader) if train_dataloader else None

        self.logger = logger

        self.hp = self.get_default_hp()
        if surrogate_configs is not None:
            # fill hyperparameters not given in surrogate_configs with default hyperparameter values.
            self.hp = {**self.hp, **surrogate_configs}
        # make hyperparameters callable by dot notation
        self.hp = SimpleNamespace(**self.hp)

        self.nr_features = nr_features
        self.seed = self.hp.seed
        self.set_seed(self.seed)

    def get_hyperparameters(self):
        return vars(self.hp)

    def get_default_hp(self) -> Dict[str, Any]:
        raise NotImplementedError

    def set_seed(self, seed):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def set_dataloader(self, train_dataloader):
        assert train_dataloader is not None
        self.train_dataloader = train_dataloader
        self.train_dataloader_it = iter(self.train_dataloader)

    def get_dataloader(self, dataset, batch_size, seed=None):
        # make the training dataset here
        if seed is None:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
            )
        else:
            def seed_worker(worker_id):
                worker_seed = torch.initial_seed() % 2 ** 32
                np.random.seed(worker_seed)
                random.seed(worker_seed)

            g = torch.Generator()
            g.manual_seed(int(seed))

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                worker_init_fn=seed_worker,
                generator=g,
                shuffle=True,
            )
        dataloader = WrappedDataLoader(dataloader, dev=self.device)
        return dataloader
