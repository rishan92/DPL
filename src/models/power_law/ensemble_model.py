import numpy as np
from typing import List, Type
from src.models.base.base_pytorch_module import BasePytorchModule
from src.data_loader.surrogate_data_loader import SurrogateDataLoader
from types import SimpleNamespace
import torch.nn as nn
from src.utils.utils import get_class, classproperty
import global_variables as gv
import wandb
from pathlib import Path

from src.models.power_law.power_law_model import PowerLawModel


class EnsembleModel(BasePytorchModule):
    _instantiated_count = 0

    def __init__(self, nr_features, total_budget, surrogate_budget, seed=None, checkpoint_path: Path = '.'):
        super().__init__(nr_features=nr_features, seed=seed, checkpoint_path=checkpoint_path)

        self.total_budget = total_budget
        self.surrogate_budget = surrogate_budget
        self.exploration_exploitation_strategy_budget = int(total_budget / 3)

        model_class: PowerLawModel = get_class("src/models/power_law", self.meta.model_class_name)

        if self.meta.flip_batch_norm:
            if hasattr(model_class.meta, 'use_batch_norm'):
                if EnsembleModel._instantiated_count % 2 == 1:
                    model_class.meta.use_batch_norm = False
                else:
                    model_class.meta.use_batch_norm = True

        if self.exploration_exploitation_strategy_budget < self.surrogate_budget:
            if self.meta.exploration_exploitation_strategy == "batch_norm":
                if hasattr(model_class.meta, 'use_batch_norm'):
                    model_class.meta.use_batch_norm = False

        self.model_instances: List[Type[model_class]] = [model_class] * self.meta.ensemble_size

        # set a seed already, so that it is deterministic when
        # generating the seeds of the ensemble
        self.model_seeds = np.random.choice(100, self.meta.ensemble_size, replace=False)

        # Where the models of the ensemble will be stored
        self.models = nn.ModuleList([])

        self.model_train_dataloaders = [None] * self.meta.ensemble_size
        self.model_val_dataloaders = [None] * self.meta.ensemble_size

        for i in range(self.meta.ensemble_size):
            self.models.append(
                self.model_instances[i](
                    nr_features=self.nr_features,
                    max_instances=self.meta.ensemble_size,
                    checkpoint_path=self.checkpoint_path,
                    seed=self.model_seeds[i]
                )
            )

        EnsembleModel._instantiated_count += 1

    def set_dataloader(self, train_dataloader=None, val_dataloader=None):
        assert train_dataloader is not None or val_dataloader is not None
        self.train_dataloader = train_dataloader if train_dataloader else None
        self.val_dataloader = val_dataloader if val_dataloader else None
        resample_split = hasattr(self.meta, 'resample_split') and self.meta.resample_split
        for i in range(self.meta.ensemble_size):
            if train_dataloader:
                model_train_dataloader = train_dataloader.make_dataloader(seed=self.model_seeds[i],
                                                                          resample_split=resample_split)
                self.model_train_dataloaders[i] = model_train_dataloader

            if val_dataloader:
                model_val_dataloader = val_dataloader.make_dataloader(seed=self.model_seeds[i])
                self.model_val_dataloaders[i] = model_val_dataloader

    @staticmethod
    def get_default_meta():
        hp = {
            'model_class_name': 'TargetSpaceComplex3PowerLawModel',
            # 'ConditionedPowerLawModel', # 'ComplexPowerLawModel',  # 'TargetSpaceComplex3PowerLawModel',
            'ensemble_size': 5,
            'nr_epochs': 250,
            'refine_nr_epochs': 20,
            'batch_size': 64,
            'refine_batch_size': 64,
            'exploration_exploitation_strategy': None,  # 'batch_norm',
            'flip_batch_norm': False,
            'use_resampling': False,
            'resample_split': 1.0,
        }
        return hp

    @classmethod
    def set_meta(cls, config=None, **kwargs):
        config = {} if config is None else config
        default_meta = cls.get_default_meta()
        meta = {**default_meta, **config}
        model_class = get_class("src/models/power_law", meta['model_class_name'])
        model_config = model_class.set_meta(config.get("model", None))
        meta['model'] = model_config
        cls.meta = SimpleNamespace(**meta)
        return meta

    def predict(self, test_data, **kwargs):
        self.eval()
        all_predictions = []
        predict_infos = []

        for model in self.models:
            predictions, predict_info = model.predict(test_data)
            all_predictions.append(predictions.detach().cpu().numpy())
            predict_infos.append(predict_info)

        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0)

        predict_infos = predict_infos[0]
        predict_infos = {key: value.detach().to('cpu').numpy() for key, value in predict_infos.items()}

        return mean_predictions, std_predictions, predict_infos

    def train_loop(self, train_dataset, should_refine=False, reset_optimizer=False, last_sample=None, val_dataset=None,
                   **kwargs):
        if should_refine:
            nr_epochs = self.meta.refine_nr_epochs
            batch_size = self.meta.refine_batch_size
            should_weight_last_sample = True
        else:
            nr_epochs = self.meta.nr_epochs
            batch_size = self.meta.batch_size
            should_weight_last_sample = False

        model_device = next(self.parameters()).device

        use_resampling = hasattr(self.meta, 'use_resampling') and self.meta.use_resampling

        # make the training dataloader
        train_dataloader = SurrogateDataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, seed=self.seed, dev=model_device,
            should_weight_last_sample=should_weight_last_sample, last_sample=last_sample, use_resampling=use_resampling
            # drop_last=train_dataset.X.shape[0] > batch_size and train_dataset.X.shape[0] % batch_size < 2
        )

        # make the validation dataloader
        val_dataloader = None
        if val_dataset:
            val_dataloader = SurrogateDataLoader(
                dataset=val_dataset, batch_size=512, shuffle=False, seed=self.seed, dev=model_device,
                should_weight_last_sample=should_weight_last_sample, last_sample=last_sample,
                # drop_last=train_dataset.X.shape[0] > batch_size and train_dataset.X.shape[0] % batch_size < 2
            )

        self.set_dataloader(train_dataloader=train_dataloader, val_dataloader=val_dataloader)

        self.set_seed(self.seed)
        self.train()

        for model, model_train_dataloader, model_val_dataloader, seed in zip(self.models,
                                                                             self.model_train_dataloaders,
                                                                             self.model_val_dataloaders,
                                                                             self.model_seeds):
            self.logger.info(f'Started training model with index: {model.instance_id}')
            model.train_loop(
                nr_epochs=nr_epochs,
                train_dataloader=model_train_dataloader,
                val_dataloader=model_val_dataloader,
                reset_optimizer=reset_optimizer,
                **kwargs
            )

        return None, None

    def training_step(self):
        model_loss = []
        for model in self.models:
            loss = model.train_step()
            model_loss.append(loss)

        return model_loss[0]

    def set_target_normalization_inverse_function(self, fn, std_fn=None):
        for model in self.models:
            model.set_target_normalization_inverse_function(fn=fn, std_fn=std_fn)

    @classproperty
    def meta_use_learning_curve(cls):
        model_class = get_class("src/models/power_law", cls.meta.model_class_name)
        return model_class.meta_use_learning_curve

    @classproperty
    def meta_use_learning_curve_mask(cls):
        model_class = get_class("src/models/power_law", cls.meta.model_class_name)
        return model_class.meta_use_learning_curve_mask

    @classproperty
    def meta_output_act_func(cls):
        model_class = get_class("src/models/power_law", cls.meta.model_class_name)
        return model_class.meta_output_act_func

    @classproperty
    def meta_cnn_kernel_size(cls):
        model_class = get_class("src/models/power_law", cls.meta.model_class_name)
        return model_class.meta_cnn_kernel_size

    @classproperty
    def meta_use_sample_weights(cls):
        model_class = get_class("src/models/power_law", cls.meta.model_class_name)
        return model_class.meta_use_sample_weights

    @classproperty
    def meta_use_sample_weight_by_budget(cls):
        model_class = get_class("src/models/power_law", cls.meta.model_class_name)
        return model_class.meta_use_sample_weight_by_budget

    @classproperty
    def meta_sample_weight_by_budget_strategy(cls):
        model_class = get_class("src/models/power_law", cls.meta.model_class_name)
        return model_class.meta_sample_weight_by_budget_strategy

    @classproperty
    def meta_use_sample_weight_by_label(cls):
        model_class = get_class("src/models/power_law", cls.meta.model_class_name)
        return model_class.meta_use_sample_weight_by_label

    @property
    def has_batchnorm_layers(self):
        return self.model[0].has_batchnorm_layers

    def reset(self):
        if gv.IS_WANDB and gv.PLOT_GRADIENTS:
            wandb.unwatch()
        for model in self.models:
            model.reset()
