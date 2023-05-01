from copy import deepcopy
import os
from typing import Dict, Tuple, Any, Optional
from numpy.typing import NDArray

import numpy as np
import torch
from loguru import logger

import gpytorch
import wandb
from gpytorch.constraints import Interval, GreaterThan, LessThan
from types import SimpleNamespace

from src.models.deep_kernel_learning.feature_extractor import FeatureExtractor
from src.models.deep_kernel_learning.gp_regression_model import GPRegressionModel
from src.models.base.base_pytorch_module import BasePytorchModule
from src.dataset.tabular_dataset import TabularDataset
from src.utils.utils import get_class_from_package, get_class_from_packages
import src.models.deep_kernel_learning
from src.models.base.meta import Meta
from src.utils.utils import get_class, classproperty


class DyHPOModel(BasePytorchModule):
    """
    The DyHPO DeepGP model.
    """
    _global_epoch = 0
    _training_errors = 0

    def __init__(
        self,
        nr_features,
        dataset_name: str = 'unknown',
        checkpoint_path: str = '.',
        seed=None
    ):
        """
        The constructor for the DyHPO model.

        Args:
            configuration: The configuration to be used
                for the different parts of the surrogate.
            device: The device where the experiments will be run on.
            dataset_name: The name of the dataset for the current run.
            output_path: The path where the intermediate/final results
                will be stored.
            seed: The seed that will be used to store the checkpoint
                properly.
        """
        super().__init__(nr_features=nr_features, seed=seed)
        self.feature_extractor_class = get_class_from_package(src.models.deep_kernel_learning,
                                                              self.meta.feature_class_name)
        self.gp_class = get_class_from_package(src.models.deep_kernel_learning,
                                               self.meta.gp_class_name)
        self.likelihood_class = get_class_from_packages([gpytorch.likelihoods, src.models.deep_kernel_learning],
                                                        self.meta.likelihood_class_name)
        self.mll_criterion_class = get_class_from_packages([gpytorch.mlls, src.models.deep_kernel_learning],
                                                           self.meta.mll_loss_function)
        self.power_law_criterion_class = get_class_from_packages([torch.nn, src.models.deep_kernel_learning],
                                                                 self.meta.power_law_loss_function)

        self.feature_extractor: FeatureExtractor = self.feature_extractor_class(nr_features=nr_features)
        self.feature_output_size = self.get_feature_extractor_output_size(nr_features=nr_features)

        self.early_stopping_patience = self.meta.nr_patience_epochs
        self.seed = seed

        if self.meta.noise_lower_bound is not None and self.meta.noise_upper_bound is not None:
            self.noise_constraint = Interval(lower_bound=self.meta.noise_lower_bound,
                                             upper_bound=self.meta.noise_upper_bound)
        elif self.meta.noise_lower_bound is not None:
            self.noise_constraint = GreaterThan(lower_bound=self.meta.noise_lower_bound)
        elif self.meta.noise_upper_bound is not None:
            self.noise_constraint = LessThan(upper_bound=self.meta.noise_upper_bound)
        else:
            self.noise_constraint = None

        train_size = self.feature_output_size
        train_x = torch.ones(train_size, train_size)
        train_y = torch.ones(train_size)

        self.likelihood: gpytorch.likelihoods.Likelihood = self.likelihood_class(noise_constraint=self.noise_constraint)
        self.model: gpytorch.models.GP = self.gp_class(train_x=train_x, train_y=train_y, likelihood=self.likelihood)
        self.mll_criterion: gpytorch.mlls.MarginalLogLikelihood = self.mll_criterion_class(self.likelihood, self.model)
        self.power_law_criterion = self.power_law_criterion_class()

        self.set_optimizer()

        self.logger = logger

        self.checkpoint_path = checkpoint_path

        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.checkpoint_file = os.path.join(
            self.checkpoint_path,
            'checkpoint.pth'
        )

    @staticmethod
    def get_default_meta(**kwargs) -> Dict[str, Any]:
        hp = {
            'batch_size': 64,
            'nr_patience_epochs': 10,
            'learning_rate': 0.001,
            'nr_epochs': 1000,
            'refine_nr_epochs': 50,
            'feature_class_name': 'FeatureExtractorDYHPO',
            # 'FeatureExtractor',  #  'FeatureExtractorPowerLaw',
            'gp_class_name': 'GPRegressionModel',
            # 'GPRegressionPowerLawMeanModel',  #  'GPRegressionModel'
            'likelihood_class_name': 'GaussianLikelihood',
            'mll_loss_function': 'ExactMarginalLogLikelihood',
            'power_law_loss_function': 'MSELoss',
            'power_law_loss_factor': 0,
            'noise_lower_bound': None,  # 1e-4,  #
            'noise_upper_bound': None,  # 1e-1,  #
            'optimizer': 'Adam',
        }

        return hp

    @classmethod
    def set_meta(cls, config=None, **kwargs):
        config = {} if config is None else config
        default_meta = cls.get_default_meta()
        meta = {**default_meta, **config}
        feature_model_class = get_class("src/models/deep_kernel_learning", meta['feature_class_name'])
        feature_model_config = feature_model_class.set_meta(config.get("feature_model", None))
        meta['feature_model'] = feature_model_config
        cls.meta = SimpleNamespace(**meta)
        return meta

    def set_optimizer(self):
        optimizer_class = get_class_from_package(torch.optim, self.meta.optimizer)
        self.optimizer = optimizer_class([
            {'params': self.model.parameters(), 'lr': self.meta.learning_rate},
            {'params': self.feature_extractor.parameters(), 'lr': self.meta.learning_rate}],
        )

    def get_model_likelihood_mll(
        self,
        train_size: int,
    ) -> Tuple[gpytorch.models.GP, gpytorch.likelihoods.Likelihood, gpytorch.mlls.MarginalLogLikelihood]:
        """
        Called when the surrogate is first initialized or restarted.

        Args:
            train_size: The size of the current training set.

        Returns:
            model, likelihood, mll - The GP model, the likelihood and
                the marginal likelihood.
        """
        train_x = torch.ones(train_size, train_size)
        train_y = torch.ones(train_size)

        if self.meta.noise_lower_bound is not None and self.meta.noise_upper_bound is not None:
            noise_constraint = Interval(lower_bound=self.meta.noise_lower_bound,
                                        upper_bound=self.meta.noise_upper_bound)
        elif self.meta.noise_lower_bound is not None:
            noise_constraint = GreaterThan(lower_bound=self.meta.noise_lower_bound)
        elif self.meta.noise_upper_bound is not None:
            noise_constraint = LessThan(upper_bound=self.meta.noise_upper_bound)
        else:
            noise_constraint = None

        likelihood = self.likelihood_class(noise_constraint=noise_constraint)
        model = self.gp_class(train_x=train_x, train_y=train_y, likelihood=likelihood)
        mll = self.mll_class(likelihood, model)

        return model, likelihood, mll

    def train_loop(self, train_dataset: TabularDataset, should_refine: bool = False, load_checkpoint: bool = False,
                   reset_optimizer=False, **kwargs):
        """
        Train the surrogate model.

        Args:
            train_dataset: A Dataset which has the training examples, training features,
                training budgets and in the end the training curves.
            load_checkpoint: A flag whether to load the state from a previous checkpoint,
                or whether to start from scratch.
        """
        if should_refine:
            nr_epochs = self.meta.refine_nr_epochs
        else:
            nr_epochs = self.meta.nr_epochs

        self.set_seed(self.seed)
        self.train()
        if load_checkpoint:
            try:
                self.load_checkpoint()
            except FileNotFoundError:
                self.logger.error(f'No checkpoint file found at: {self.checkpoint_file}'
                                  f'Training the GP from the beginning')

        if reset_optimizer:
            self.set_optimizer()

        model_device = next(self.parameters()).device
        train_dataset.to(model_device)

        X_train = train_dataset.X
        train_budgets = train_dataset.budgets
        train_curves = train_dataset.curves
        y_train = train_dataset.Y

        initial_state = self.get_state()
        training_errored = False

        # where the mean squared error will be stored
        # when predicting on the train set
        mse = 0.0

        nr_examples_batch = X_train.size(dim=0)
        # if only one example in the batch, skip the batch.
        # Otherwise, the code will fail because of batchnorm
        if nr_examples_batch == 1:
            return

        for epoch_nr in range(0, nr_epochs):
            DyHPOModel._global_epoch += 1

            # Zero backprop gradients
            self.optimizer.zero_grad()

            projected_x, _ = self.feature_extractor(X_train, train_budgets, train_curves)
            self.model.set_train_data(projected_x, y_train, strict=False)
            output: gpytorch.ExactMarginalLogLikelihood = self.model(projected_x)

            try:
                # Calc loss and backprop derivatives
                mll_loss = -self.mll_criterion(output, self.model.train_targets)

                prediction: gpytorch.distributions.Distribution = self.likelihood(output)
                power_law_output = projected_x[:, -1]
                mean_prediction = prediction.mean
                power_law_loss = self.power_law_criterion(mean_prediction, power_law_output)

                loss = mll_loss + self.meta.power_law_loss_factor * power_law_loss

                mae = gpytorch.metrics.mean_absolute_error(prediction, self.model.train_targets)
                power_law_mae = gpytorch.metrics.mean_absolute_error(prediction, power_law_output.detach())

                mll_loss_value = mll_loss.detach().to('cpu').item()
                power_law_loss_value = power_law_loss.detach().to('cpu').item()
                loss_value = loss.detach().to('cpu').item()
                mae_value = mae.detach().to('cpu').item()
                power_law_mae_value = power_law_mae.detach().to('cpu').item()

                self.logger.debug(
                    f'Epoch {epoch_nr} - MAE {mae_value}, '
                    f'Loss: {loss_value}, '
                    f'lengthscale: {self.model.covar_module.base_kernel.lengthscale.item()}, '
                    f'noise: {self.model.likelihood.noise.item()}, '
                )

                wandb.log({
                    "surrogate/dyhpo/training_loss": loss_value,
                    "surrogate/dyhpo/training_mll_loss": mll_loss_value,
                    "surrogate/dyhpo/training_power_law_loss": power_law_loss_value,
                    "surrogate/dyhpo/training_MAE": mae_value,
                    "surrogate/dyhpo/training_power_law_MAE": power_law_mae_value,
                    "surrogate/dyhpo/training_lengthscale": self.model.covar_module.base_kernel.lengthscale.item(),
                    "surrogate/dyhpo/training_noise": self.model.likelihood.noise.item(),
                    "surrogate/dyhpo/epoch": DyHPOModel._global_epoch,
                    "surrogate/dyhpo/training_errors": DyHPOModel._training_errors,
                })

                loss.backward()
                self.optimizer.step()
            except Exception as training_error:
                self.logger.error(f'The following error happened at epoch {nr_epochs} while training: {training_error}')
                # raise training_error
                # An error has happened, trigger the restart of the optimization and restart
                # the model with default hyperparameters.
                training_errored = True
                DyHPOModel._training_errors += 1
                wandb.log({
                    f"surrogate/dyhpo/epoch": DyHPOModel._global_epoch,
                    f"surrogate/dyhpo/training_errors": DyHPOModel._training_errors,
                })
                break

        check_seed_torch = torch.random.get_rng_state().sum()
        self.logger.debug(f"end rng_state {check_seed_torch}")

        """
        # metric too high, time to restart, or we risk divergence
        if mse > 0.15:
            if not self.restart:
                self.restart = True
        """
        if training_errored:
            self.save_checkpoint(initial_state)
            self.load_checkpoint()

        return_state = -1 if training_errored else 0
        return return_state

    def predict(
        self,
        test_data: TabularDataset,
        train_data: TabularDataset
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], Optional[Dict[str, Any]]]:
        """

        Args:
            train_data: A dictionary that has the training
                examples, features, budgets and learning curves.
            test_data: Same as for the training data, but it is
                for the testing part and it does not feature labels.

        Returns:
            means, stds: The means of the predictions for the
                testing points and the standard deviations.
        """
        model_device = next(self.parameters()).device
        train_data.to(model_device)
        test_data.to(model_device)

        self.eval()

        with torch.no_grad():  # gpytorch.settings.fast_pred_var():
            projected_train_x, _ = self.feature_extractor(
                train_data.X,
                train_data.budgets,
                train_data.curves,
            )
            self.model.set_train_data(inputs=projected_train_x, targets=train_data.Y, strict=False)
            projected_test_x, test_predict_infos = self.feature_extractor(
                test_data.X,
                test_data.budgets,
                test_data.curves,
            )
            preds: gpytorch.distributions.Distribution = self.likelihood(self.model(projected_test_x))

            power_law_output = projected_test_x[:, -1]
            power_law_loss = gpytorch.metrics.mean_absolute_error(preds, power_law_output)
            power_law_loss_value = power_law_loss.detach().to('cpu').item()
            wandb.log({
                "surrogate/dyhpo/testing_power_law_MAE": power_law_loss_value
            })

        means = preds.mean.detach().to('cpu').numpy().reshape(-1, )
        stds = preds.stddev.detach().to('cpu').numpy().reshape(-1, )

        predict_infos = test_predict_infos
        if predict_infos is not None:
            predict_infos = {key: value.detach().to('cpu').numpy() for key, value in predict_infos.items()}

        return means, stds, predict_infos

    def load_checkpoint(self):
        """
        Load the state from a previous checkpoint.
        """
        checkpoint = torch.load(self.checkpoint_file)
        self.model.load_state_dict(checkpoint['gp_state_dict'])
        self.feature_extractor.load_state_dict(checkpoint['feature_extractor_state_dict'])
        self.likelihood.load_state_dict(checkpoint['likelihood_state_dict'])

    def save_checkpoint(self, state: Dict = None):
        """
        Save the given state or the current state in a
        checkpoint file.

        Args:
            state: The state to save, if none, it will
            save the current state.
        """

        if state is None:
            torch.save(
                self.get_state(),
                self.checkpoint_file,
            )
        else:
            torch.save(
                state,
                self.checkpoint_file,
            )

    def get_state(self) -> Dict[str, Dict]:
        """
        Get the current state of the surrogate.

        Returns:
            current_state: A dictionary that represents
                the current state of the surrogate model.
        """
        current_state = {
            'gp_state_dict': deepcopy(self.model.state_dict()),
            'feature_extractor_state_dict': deepcopy(self.feature_extractor.state_dict()),
            'likelihood_state_dict': deepcopy(self.likelihood.state_dict()),
        }

        return current_state

    def to(self, dev):
        self.feature_extractor.to(dev)
        self.model.to(dev)
        self.likelihood.to(dev)

    def train(self, **kwargs):
        self.feature_extractor.train(**kwargs)
        self.model.train(**kwargs)
        self.likelihood.train(**kwargs)

    def eval(self):
        self.feature_extractor.eval()
        self.model.eval()
        self.likelihood.eval()

    @classproperty
    def use_learning_curve(cls):
        model_class = get_class("src/models/deep_kernel_learning", cls.meta.feature_class_name)
        return model_class.use_learning_curve

    @classproperty
    def use_learning_curve_mask(cls):
        model_class = get_class("src/models/deep_kernel_learning", cls.meta.feature_class_name)
        return model_class.use_learning_curve_mask

    def get_feature_extractor_output_size(self, nr_features):
        valid_budget_size = 50
        # Create a dummy input with the appropriate input size
        dummy_x_input = torch.zeros(4, nr_features)
        dummy_budget_input = torch.ones(4)
        dummy_learning_curves_input = torch.zeros(4, valid_budget_size)
        self.feature_extractor.eval()
        output, _ = self.feature_extractor(dummy_x_input, dummy_budget_input, dummy_learning_curves_input)
        output_size = output.shape[-1]
        return output_size
