import random
from copy import deepcopy
import logging
import os
from typing import Dict, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch import cat
from loguru import logger

import gpytorch

from src.models.deep_kernel_learning.feature_extractor import FeatureExtractor
from src.models.deep_kernel_learning.gp_regression_model import GPRegressionModel
from src.models.power_law.base_pytorch_module import BasePytorchModule


class DyHPOModel(BasePytorchModule):
    """
    The DyHPO DeepGP model.
    """

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
        check_seed_torch = torch.random.get_rng_state().sum()
        check_seed_np = np.sum(np.random.get_state()[1])
        check_seed_random = np.sum(random.getstate()[1])
        self.hp.nr_features = nr_features
        self.feature_extractor = FeatureExtractor(self.hp)

        self.early_stopping_patience = self.hp.nr_patience_epochs
        self.seed = seed
        self.model, self.likelihood, self.mll = \
            self.get_model_likelihood_mll(
                getattr(self.hp, f'layer{self.feature_extractor.nr_layers}_units')
            )

        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': self.hp.learning_rate},
            {'params': self.feature_extractor.parameters(), 'lr': self.hp.learning_rate}],
        )

        # the number of initial points for which we will retrain fully from scratch
        # This is basically equal to the dimensionality of the search space + 1.
        self.initial_nr_points = 10
        # keeping track of the total hpo iterations. It will be used during the optimization
        # process to switch from fully training the model, to refining.
        self.iterations = 0
        # flag for when the optimization of the model should start from scratch.
        self.restart = True

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
            'nr_layers': 2,
            'layer1_units': 64,
            'layer2_units': 128,
            'cnn_nr_channels': 4,
            'cnn_kernel_size': 3,
            'batch_size': 64,
            'nr_patience_epochs': 10,
            'learning_rate': 0.001,
            'nr_epochs': 1000,
            'refine_nr_epochs': 50,
            'use_learning_curve': True,
            'use_learning_curve_mask': False,
            'feature_class_name': 'FeatureExtractor',
            'gp_class_name': 'GPRegressionModel',
        }
        return hp

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

    def restart_optimization(self):
        """
        Restart the surrogate model from scratch.
        """
        check_seed_torch = torch.random.get_rng_state().sum()
        check_seed_np = np.sum(np.random.get_state()[1])
        check_seed_random = np.sum(random.getstate()[1])
        self.feature_extractor = FeatureExtractor(self.hp)
        self.model, self.likelihood, self.mll = \
            self.get_model_likelihood_mll(
                getattr(self.hp, f'layer{self.feature_extractor.nr_layers}_units'),
            )

        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': self.hp.learning_rate},
            {'params': self.feature_extractor.parameters(), 'lr': self.hp.learning_rate}],
        )

    def get_model_likelihood_mll(
        self,
        train_size: int,
    ) -> Tuple[GPRegressionModel, gpytorch.likelihoods.GaussianLikelihood, gpytorch.mlls.ExactMarginalLogLikelihood]:
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

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GPRegressionModel(train_x=train_x, train_y=train_y, likelihood=likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        return model, likelihood, mll

    def train_loop(self, train_dataset, should_refine=False, load_checkpoint: bool = False,
                   **kwargs):
        """
        Train the surrogate model.

        Args:
            data: A dictionary which has the training examples, training features,
                training budgets and in the end the training curves.
            load_checkpoint: A flag whether to load the state from a previous checkpoint,
                or whether to start from scratch.
        """
        if should_refine:
            nr_epochs = self.hp.refine_nr_epochs
        else:
            nr_epochs = self.hp.nr_epochs

        self.set_seed(self.seed)
        self.train()
        if load_checkpoint:
            try:
                self.load_checkpoint()
            except FileNotFoundError:
                self.logger.error(f'No checkpoint file found at: {self.checkpoint_file}'
                                  f'Training the GP from the beginning')

        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters(), 'lr': self.hp.learning_rate},
            {'params': self.feature_extractor.parameters(), 'lr': self.hp.learning_rate}],
        )
        model_device = next(self.parameters()).device
        train_dataset.to(model_device)

        X_train = train_dataset.X
        train_budgets = train_dataset.budgets
        train_curves = train_dataset.curves
        y_train = train_dataset.Y

        initial_state = self.get_state()
        training_errored = False
        check_seed_torch = torch.random.get_rng_state().sum()
        # where the mean squared error will be stored
        # when predicting on the train set
        mse = 0.0

        for epoch_nr in range(0, nr_epochs):
            nr_examples_batch = X_train.size(dim=0)
            # if only one example in the batch, skip the batch.
            # Otherwise, the code will fail because of batchnorm
            if nr_examples_batch == 1:
                continue

            # Zero backprop gradients
            self.optimizer.zero_grad()

            projected_x = self.feature_extractor(X_train, train_budgets, train_curves)
            self.model.set_train_data(projected_x, y_train, strict=False)
            output = self.model(projected_x)

            try:
                # Calc loss and backprop derivatives
                loss = -self.mll(output, self.model.train_targets)
                loss_value = loss.detach().to('cpu').item()
                mse = gpytorch.metrics.mean_squared_error(output, self.model.train_targets)
                self.logger.debug(
                    f'Epoch {epoch_nr} - MSE {mse}, '
                    f'Loss: {loss_value}, '
                    f'lengthscale: {self.model.covar_module.base_kernel.lengthscale.item()}, '
                    f'noise: {self.model.likelihood.noise.item()}, '
                )
                loss.backward()
                self.optimizer.step()
            except Exception as training_error:
                self.logger.error(f'The following error happened while training: {training_error}')
                # An error has happened, trigger the restart of the optimization and restart
                # the model with default hyperparameters.
                self.restart = True
                training_errored = True
                break
        check_seed_torch = torch.random.get_rng_state().sum()
        self.logger.info(f"end rng_state {check_seed_torch}")
        """
        # metric too high, time to restart, or we risk divergence
        if mse > 0.15:
            if not self.restart:
                self.restart = True
        """
        if training_errored:
            self.save_checkpoint(initial_state)
            self.load_checkpoint()

    def predict(
        self,
        test_data,
        train_data
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        check_seed_torch = torch.random.get_rng_state().sum()
        check_seed_np = np.sum(np.random.get_state()[1])
        check_seed_random = np.sum(random.getstate()[1])
        with torch.no_grad():  # gpytorch.settings.fast_pred_var():
            projected_train_x = self.feature_extractor(
                train_data.X,
                train_data.budgets,
                train_data.curves,
            )
            self.model.set_train_data(inputs=projected_train_x, targets=train_data.Y, strict=False)
            projected_test_x = self.feature_extractor(
                test_data.X,
                test_data.budgets,
                test_data.curves,
            )
            preds = self.likelihood(self.model(projected_test_x))

        means = preds.mean.detach().to('cpu').numpy().reshape(-1, )
        stds = preds.stddev.detach().to('cpu').numpy().reshape(-1, )

        return means, stds

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
