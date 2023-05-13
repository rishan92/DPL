import torch
import torch.nn as nn
import math

from src.models.power_law.power_law_model import PowerLawModel
from src.models.layers.scaling_layer import ScalingLayer
import src.models.activation_functions as act


class TargetSpaceComplex3PowerLawModel(PowerLawModel):
    param_names = ('alphas_r', 'betas_r', 'gammas_r')

    @staticmethod
    def get_default_meta():
        hp = {
            'nr_units': 128,
            'nr_layers': 2,
            'kernel_size': 3,
            'nr_filters': 4,
            'nr_cnn_layers': 2,
            'dropout_rate': 0,
            'use_batch_norm': False,
            'use_learning_curve': False,
            'use_learning_curve_mask': False,
            'use_suggested_learning_rate': False,
            'use_sample_weights': False,
            'use_sample_weight_by_budget': True,
            'sample_weight_by_budget_strategy': None,
            'weight_regularization_factor': 0,
            'alpha_beta_constraint_factor': 0,
            'gamma_constraint_factor': 0,
            'learning_rate': 1e-3,
            'refine_learning_rate': 1e-3,
            'act_func': 'LeakyReLU',
            'last_act_func': 'Identity',
            'alpha_act_func': 'Sigmoid',
            'beta_act_func': 'Sigmoid',
            'gamma_act_func': 'Sigmoid',
            'output_act_func': None,
            'alpha_beta_is_difference': True,
            'use_gamma_constraint': True,
            'loss_function': 'L1Loss',
            'optimizer': 'Adam',
            'learning_rate_scheduler': None,
            # 'CosineAnnealingLR' 'LambdaLR' 'OneCycleLR' 'ExponentialLR'
            'learning_rate_scheduler_args': {
                'total_iters_factor': 1.0,
                'eta_min': 1e-6,
                'max_lr': 1e-4,
                'refine_max_lr': 1e-3,
                'exp_min': 1e-6,
                'refine_exp_min': 1e-6,
            },
            'activate_early_stopping': False,
            'early_stopping_it': 0,
            'use_scaling_layer': False,
            'scaling_layer_bias_values': [0.0, -0.2, 0.0]  # [0, 0, 1.17125493757],
        }
        return hp

    def get_linear_net(self):
        layers = []
        # adding one since we concatenate the features with the budget
        nr_initial_features = self.nr_features
        if self.meta.use_learning_curve:
            nr_initial_features = self.nr_features + self.meta.nr_filters

        layers.append(nn.Linear(nr_initial_features, self.meta.nr_units))
        if hasattr(self.meta, 'use_batch_norm') and self.meta.use_batch_norm:
            layers.append(nn.BatchNorm1d(self.meta.nr_units))
        layers.append(self.act_func)
        if hasattr(self.meta, 'dropout_rate') and self.meta.dropout_rate != 0:
            layers.append(nn.Dropout(self.meta.dropout_rate))

        for i in range(2, self.meta.nr_layers + 1):
            layers.append(nn.Linear(self.meta.nr_units, self.meta.nr_units))
            if hasattr(self.meta, 'use_batch_norm') and self.meta.use_batch_norm:
                layers.append(nn.BatchNorm1d(self.meta.nr_units))
            layers.append(self.act_func)
            if hasattr(self.meta, 'dropout_rate') and self.meta.dropout_rate != 0:
                layers.append(nn.Dropout(self.meta.dropout_rate))

        last_layer = nn.Linear(self.meta.nr_units, 3)
        layers.append(last_layer)

        if hasattr(self.meta, "use_scaling_layer") and self.meta.use_scaling_layer:
            bias_values = None
            if hasattr(self.meta, "scaling_layer_bias_values") and self.meta.scaling_layer_bias_values:
                bias_values = self.meta.scaling_layer_bias_values
            scaling_layer = ScalingLayer(in_features=3, bias_values=bias_values)
            layers.append(scaling_layer)

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
        y1 = x[:, 1]
        y2 = x[:, 2]

        alphas = self.alpha_act_func(alphas)
        y1 = self.beta_act_func(y1)
        y2 = self.gamma_act_func(y2)

        if hasattr(self.meta, 'alpha_beta_is_difference') and self.meta.alpha_beta_is_difference:
            y2 = y1 * y2
            # min_val = torch.min(y1, y2)
            # max_val = torch.max(y1, y2)
            # y1 = max_val
            # y2 = min_val

        if hasattr(self.meta, 'use_gamma_constraint') and self.meta.use_gamma_constraint:
            alphas = alphas * y2
            # min_val = torch.min(y2, alphas)
            # max_val = torch.max(y2, alphas)
            # y2 = max_val
            # alphas = min_val

        val = ((y2 - alphas) / (y1 - alphas + torch.tensor(1e-4))) + torch.tensor(1e-4)

        abs_val = torch.abs(val)
        log_abs_val = torch.log(abs_val)

        # # Calculate the angle (imaginary part)
        # angle = torch.atan2(torch.tensor(0.0), val)
        #
        # log_val = torch.complex(log_abs_val, angle)

        gammas = log_abs_val / torch.log(torch.tensor(1 / 51))

        betas = y2 - alphas

        output = torch.add(
            alphas,
            torch.mul(
                betas,
                torch.pow(
                    predict_budgets,
                    torch.mul(gammas, -1)
                )
            ),
        )
        # output = output_complex.real
        if self.output_act_func and self.training:
            output = self.output_act_func(output)

        info = {
            'alpha': alphas,
            'beta': betas,
            'gamma': gammas,
            'pl_output': output,
        }

        return output, info
