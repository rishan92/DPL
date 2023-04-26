import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
import wandb

from src.models.power_law.power_law_model import PowerLawModel


class ConstrainedPowerLawModel(PowerLawModel):
    @staticmethod
    def get_default_meta():
        hp = {
            'nr_units': 128,
            'nr_layers': 2,
            'kernel_size': 3,
            'nr_filters': 4,
            'nr_cnn_layers': 2,
            'use_learning_curve': False,
            'use_learning_curve_mask': False,
            'learning_rate': 0.001,
            'act_func': 'LeakyReLU',
            'alpha_act_func': 'Identity',
            'beta_act_func': 'Abs',
            'gamma_act_func': 'Abs',
            'loss_function': 'L1Loss',
            'optimizer': 'Adam',
            'activate_early_stopping': False,
            'early_stopping_it': 0,
        }
        return hp

    def get_linear_net(self):
        layers = []
        # adding one since we concatenate the features with the budget
        nr_initial_features = self.nr_features
        if self.meta.use_learning_curve:
            nr_initial_features = self.nr_features + self.meta.nr_filters

        layers.append(nn.Linear(nr_initial_features, self.meta.nr_units))
        layers.append(self.act_func)

        for i in range(2, self.meta.nr_layers + 1):
            layers.append(nn.Linear(self.meta.nr_units, self.meta.nr_units))
            layers.append(self.act_func)

        last_layer = nn.Linear(self.meta.nr_units, 3)
        layers.append(last_layer)

        net = torch.nn.Sequential(*layers)
        return net

    def forward(self, batch):
        """
        Args:
            x: torch.Tensor
                The examples.
            predict_budgets: torch.Tensor
                The budgets for which the performance will be predicted for the
                hyperparameter configurations.
            evaluated_budgets: torch.Tensor
                The budgets for which the hyperparameter configurations have been
                evaluated so far.
            learning_curves: torch.Tensor
                The learning curves for the hyperparameter configurations.
        """
        x, predict_budgets, learning_curves = batch

        x = self.linear_net(x)
        alphas = x[:, 0]
        betas = x[:, 1]
        gammas = x[:, 2]

        constrained_alpha = self.alpha_act_func(alphas)
        constrained_beta = self.beta_act_func(betas)
        constrained_gamma = self.gamma_act_func(gammas)

        # constrain_sum = constrained_alpha + constrained_beta
        # lower_ab_constraint = torch.clamp(0 - constrain_sum, min=0)
        # lower_a_constraint = torch.clamp(-1 * constrained_alpha, min=0)
        # # lower_b_constraint = torch.clamp(-1 * constrained_beta, min=0)
        #
        # lower_constraint = torch.max(lower_ab_constraint, lower_a_constraint)
        #
        # # a = torch.sum(lower_constraint)
        # # if a > 0:
        # #     print(f"adjustment {a}")
        #
        # constrained_alpha = constrained_alpha + lower_constraint
        # constrained_beta = constrained_beta + lower_constraint
        #
        # constrain_sum = constrained_alpha + constrained_beta
        # upper_constraint = torch.clamp(constrain_sum - 1, min=0)
        #
        # # a = torch.sum(upper_constraint)
        # # if a > 0:
        # #     print(f"adjustment {a}")
        #
        # constrained_alpha = constrained_alpha / (1 + upper_constraint)
        # constrained_beta = constrained_beta / (1 + upper_constraint)

        # constrained_beta = 1 - constrained_beta
        # constrained_beta = self.beta_act_func(constrained_beta)

        budget_lower_limit = torch.tensor(1 / 51)
        budget_upper_limit = torch.tensor(1)
        # budget_lower_limit = torch.tensor(1)
        # budget_upper_limit = torch.tensor(51)
        # factor = torch.log(constrained_beta) / torch.log(budget_limit)
        y_limit = torch.tensor(0.001)
        # factor = (torch.log(constrained_beta) - torch.log(y_limit)) / torch.log(budget_limit)
        # factor = (-1 * torch.log(y_limit)) / torch.log(budget_upper_limit)
        # constrained_gamma = factor * (1 - constrained_gamma)
        # constrained_gamma = self.gamma_act_func(constrained_gamma)

        # constrained_gamma = constrained_gamma * factor

        output = torch.add(
            constrained_alpha,
            torch.mul(
                constrained_beta,
                torch.pow(
                    predict_budgets,
                    torch.mul(constrained_gamma, -1)
                )
            ),
        )

        # start_output = torch.add(
        #     constrained_alpha,
        #     torch.mul(
        #         constrained_beta,
        #         torch.pow(
        #             budget_lower_limit,
        #             torch.mul(constrained_gamma, -1)
        #         )
        #     ),
        # )
        # end_output = torch.add(
        #     constrained_alpha,
        #     torch.mul(
        #         constrained_beta,
        #         torch.pow(
        #             budget_upper_limit,
        #             torch.mul(constrained_gamma, -1)
        #         )
        #     ),
        # )
        # print(f"start {start_output}")
        # print(f"end {end_output}")

        return output  # , constrained_alpha, constrained_beta, constrained_gamma
