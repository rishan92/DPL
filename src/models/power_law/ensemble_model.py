from src.models.power_law.conditioned_power_law_model import ConditionedPowerLawModel
import numpy as np
from copy import deepcopy
from typing import List, Tuple, Dict, Optional, Any, Type
from src.models.power_law.base_pytorch_module import BasePytorchModule
import torch
from src.data_loader.surrogate_data_loader import SurrogateDataLoader
from src.utils.utils import merge_dicts, get_class
from types import SimpleNamespace
import torch.nn as nn


class EnsembleModel(BasePytorchModule):
    def __init__(self, nr_features, seed=None, checkpoint_path: str = '.'):
        super().__init__(nr_features=nr_features, seed=seed, checkpoint_path=checkpoint_path)

        model_class = get_class("src/models/power_law", self.hp.model_class)
        self.model_instances: List[Type[model_class]] = [model_class] * self.hp.ensemble_size

        # set a seed already, so that it is deterministic when
        # generating the seeds of the ensemble
        self.model_seeds = np.random.choice(100, self.hp.ensemble_size, replace=False)

        # Where the models of the ensemble will be stored
        self.models = nn.ModuleList([])

        self.model_train_dataloaders = [None] * self.hp.ensemble_size

        for i in range(self.hp.ensemble_size):
            self.models.append(
                self.model_instances[i](
                    nr_features=self.nr_features,
                    max_instances=self.hp.ensemble_size,
                    train_dataloader=self.model_train_dataloaders[i],
                    checkpoint_path=self.checkpoint_path,
                    seed=self.model_seeds[i]
                )
            )

    def set_dataloader(self, train_dataloader):
        if train_dataloader is not None:
            self.train_dataloader = train_dataloader
            for i in range(self.hp.ensemble_size):
                model_train_dataloader = train_dataloader.make_dataloader(seed=self.model_seeds[i])
                self.model_train_dataloaders[i] = model_train_dataloader

    @staticmethod
    def get_default_meta():
        hp = {
            'model_class': 'ConditionedPowerLawModel',
            'ensemble_size': 5,
            'nr_epochs': 250,
            'refine_nr_epochs': 20,
            'batch_size': 64,
            'refine_batch_size': 64,
            'predict_mode': 'end_budget',  # 'next_budget'
            'curve_size_mode': 'fixed',  # 'variable'
        }
        return hp

    @classmethod
    def set_meta(cls, config=None):
        config = {} if config is None else config
        default_meta = cls.get_default_meta()
        meta = {**default_meta, **config}
        model_class = get_class("src/models/power_law", meta['model_class'])
        model_config = model_class.set_meta(config.get("model", None))
        meta['model'] = model_config
        cls.meta = SimpleNamespace(**meta)
        return meta

    def predict(self, test_data, **kwargs):
        self.eval()
        all_predictions = []

        for model in self.models:
            predictions = model.predict(test_data)
            all_predictions.append(predictions.detach().cpu().numpy())

        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0)

        return mean_predictions, std_predictions

    def train_loop(self, train_dataset, should_refine=False, reset_optimizer=False, last_sample=None, **kwargs):
        if should_refine:
            nr_epochs = self.hp.refine_nr_epochs
            batch_size = self.hp.refine_batch_size
            should_weight_last_sample = True
        else:
            nr_epochs = self.hp.nr_epochs
            batch_size = self.hp.batch_size
            should_weight_last_sample = False

        model_device = next(self.parameters()).device
        # make the training dataloader
        train_dataloader = SurrogateDataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, seed=self.seed, dev=model_device,
            should_weight_last_sample=should_weight_last_sample, last_sample=last_sample,
            # drop_last=train_dataset.X.shape[0] > batch_size and train_dataset.X.shape[0] % batch_size < 2
        )
        # initial dataloader is discarded
        self.set_dataloader(train_dataloader=train_dataloader)

        self.set_seed(self.seed)
        self.train()

        for model, model_dataloader, seed in zip(self.models, self.model_train_dataloaders, self.model_seeds):
            self.logger.info(f'Started training model with index: {model.instance_id}')
            model.train_loop(nr_epochs=nr_epochs, train_dataloader=model_dataloader, reset_optimizer=reset_optimizer)

    def training_step(self):
        model_loss = []
        for model in self.models:
            loss = model.train_step()
            model_loss.append(loss)

        return model_loss[0]
