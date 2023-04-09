import torch
import torch.nn as nn
from models.conditioned_power_law_model import ConditionedPowerLawModel
import numpy as np
import random
from typing import List, Tuple, Any, Type
from loguru import logger
from torch.utils.data import DataLoader
from copy import deepcopy
from data_loader.tabular_data_loader import WrappedDataLoader
from models.base_pytorch_module import BasePytorchModule


class EnsembleModel(BasePytorchModule):
    def __init__(self, nr_features, train_dataloader=None, surrogate_configs=None):
        super().__init__(nr_features=nr_features, train_dataloader=train_dataloader,
                         surrogate_configs=surrogate_configs)

        self.model_instances: List[Type[ConditionedPowerLawModel]] = [ConditionedPowerLawModel] * self.hp.ensemble_size

        # set a seed already, so that it is deterministic when
        # generating the seeds of the ensemble
        self.set_seed(self.hp.seed)
        self.model_seeds = np.random.choice(100, self.hp.ensemble_size, replace=False)

        # Where the models of the ensemble will be stored
        self.models: List[ConditionedPowerLawModel] = []

        self.model_train_dataloaders = [None] * self.hp.ensemble_size
        self.set_dataloader(train_dataloader=train_dataloader)

        if surrogate_configs:
            model_config = deepcopy(surrogate_configs)
        else:
            model_config = {}
        for i in range(self.hp.ensemble_size):
            model_config['seed'] = self.model_seeds[i]
            self.models.append(
                self.model_instances[i](
                    nr_features=self.nr_features,
                    train_dataloader=self.model_train_dataloaders[i],
                    surrogate_configs=model_config
                )
            )

    def set_dataloader(self, train_dataloader):
        if train_dataloader is not None:
            for i in range(self.hp.ensemble_size):
                model_train_dataloader = train_dataloader.make_dataloader(seed=self.model_seeds[i])
                self.model_train_dataloaders[i] = model_train_dataloader

    def get_default_hp(self):
        hp = {
            'seed': 0,
            'ensemble_size': 5,
        }
        return hp

    def forward(self, x):
        # configurations, budgets, network_real_budgets, hp_curves = x
        all_predictions = []

        for model in self.models:
            predictions = model(x)
            all_predictions.append(predictions.detach().cpu().numpy())

        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0)

        return mean_predictions, std_predictions

    def train_loop(self, nr_epochs, train_dataloader=None, reset_optimizer=False):
        # initial dataloader is discarded
        self.set_dataloader(train_dataloader=train_dataloader)

        self.set_seed(self.seed)

        for model, model_dataloader, seed in zip(self.models, self.model_train_dataloaders, self.model_seeds):
            self.logger.info(f'Started training model with index: {model.instance_id}')
            model.train_loop(nr_epochs=nr_epochs, train_dataloader=model_dataloader, reset_optimizer=reset_optimizer)

    def training_step(self):
        model_loss = []
        for model in self.models:
            loss = model.train_step()
            model_loss.append(loss)

        return model_loss[0]
