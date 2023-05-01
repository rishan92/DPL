import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
import wandb
import math

from src.models.power_law.power_law_model import PowerLawModel
from .scaling_layer import ScalingLayer


class TargetSpaceComplex6PowerLawModel(PowerLawModel):
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
            'last_act_func': 'Sigmoid',
            'loss_function': 'L1Loss',
            'optimizer': 'Adam',
            'activate_early_stopping': False,
            'early_stopping_it': 0,
            'use_scaling_layer': False,
            'scaling_layer_bias_values': [0, 0, math.log(0.01) / math.log(1 / 51)]  # [0, 0, 1.17125493757]
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

        last_layer = nn.Linear(self.meta.nr_units, 4)
        layers.append(last_layer)

        if hasattr(self.meta, "use_scaling_layer") and self.meta.use_scaling_layer:
            bias_values = None
            if hasattr(self.meta, "scaling_layer_bias_values") and self.meta.scaling_layer_bias_values:
                bias_values = self.meta.scaling_layer_bias_values
            scaling_layer = ScalingLayer(in_features=4, bias_values=bias_values)
            layers.append(scaling_layer)

        net = torch.nn.Sequential(*layers)
        return net

    def get_cnn_net(self):
        cnn_part = []

        cnn_part.append(
            nn.Conv1d(
                in_channels=2,
                kernel_size=(self.meta.kernel_size,),
                out_channels=self.meta.nr_filters,
            ),
        )
        for i in range(1, self.meta.nr_cnn_layers):
            cnn_part.append(self.act_func)
            cnn_part.append(
                nn.Conv1d(
                    in_channels=self.meta.nr_filters,
                    kernel_size=(self.meta.kernel_size,),
                    out_channels=self.meta.nr_filters,
                ),
            ),
        cnn_part.append(nn.AdaptiveAvgPool1d(1))

        net = torch.nn.Sequential(*cnn_part)
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

        # x = torch.cat((x, torch.unsqueeze(evaluated_budgets, 1)), dim=1)
        if self.meta.use_learning_curve:
            lc_features = self.cnn_net(learning_curves)
            # revert the output from the cnn into nr_rows x nr_kernels.
            lc_features = torch.squeeze(lc_features, 2)
            x = torch.cat((x, lc_features), dim=1)

        x = self.linear_net(x)
        alphas_r = x[:, 0]
        alphas_i = x[:, 1]
        y1 = x[:, 2]
        y2_diff = x[:, 3]

        alphas = alphas_r + 1j * alphas_i

        y1 = self.last_act_func(y1)
        y2_diff = self.last_act_func(y2_diff)

        y2 = y1 * y2_diff

        val = (y2 - alphas) / (y1 - alphas)

        abs_val = torch.abs(val)
        log_abs_val = torch.log(abs_val)

        # Calculate the angle (imaginary part)
        angle = torch.atan2(val.imag, val.real)

        log_val = torch.complex(log_abs_val, angle)

        gammas = log_val / torch.log(torch.tensor(1 / 51))

        betas = y2 - alphas

        output_complex = torch.add(
            alphas,
            torch.mul(
                betas,
                torch.pow(
                    predict_budgets,
                    torch.mul(gammas, -1)
                )
            ),
        )
        output = output_complex.real
        output_imag = output_complex.imag

        # output_r = torch.add(
        #     alphas,
        #     torch.mul(
        #         betas,
        #         torch.pow(
        #             predict_budgets,
        #             torch.mul(gammas_r, -1)
        #         )
        #     ),
        # )

        # budget_lower_limit = torch.tensor(1 / 51)
        # budget_upper_limit = torch.tensor(1)
        # constrained_alpha = alphas
        # constrained_beta = betas
        # constrained_gamma = gammas
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

        info = {
            'alpha': alphas,
            'beta': betas,
            'gamma': gammas,
            'pl_output': output,
        }

        return output, info
