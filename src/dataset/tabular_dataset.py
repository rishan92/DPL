import torch
from torch.utils.data import Dataset
from sklearn.utils import resample
import numpy as np
import random


class TabularDataset(Dataset):
    def __init__(self, X, budgets, curves=None, Y=None, use_sample_weights=False, use_sample_weight_by_budget=False,
                 weights=None):
        super().__init__()
        self.X = X
        self.Y = Y
        self.budgets = budgets
        if curves is not None:
            self.curves = curves
        else:
            self.curves = torch.tensor([0], dtype=torch.float32)
            self.curves = self.curves.expand(self.X.size(0), -1)
        self.weights = None
        self.use_sample_weights = use_sample_weights
        self.use_sample_weight_by_budget = use_sample_weight_by_budget
        self.dummy_weight = torch.tensor([1.0])
        self.weight_factor = torch.tensor(1)
        if weights is not None:
            self.weights = weights
        else:
            if use_sample_weights:
                self.weights = torch.rand((self.X.shape[0],))
                # self.weights = torch.randn((self.X.shape[0],)) * 0.1

    def get_weight(self, x, budget):
        x_index = (self.X == x).all(axis=1)
        budget_index = self.budgets == budget

        index = torch.logical_and(x_index, budget_index)
        index = index.nonzero()
        index = index[0]

        weight = self.weights[index] if self.weights is not None else self.dummy_weight
        # weight = weight.unsqueeze(0)
        return weight

    def reset_sample_weights(self, seed=None):
        if self.use_sample_weights:
            if seed:
                self.set_seed(seed)
            self.weights = torch.rand((self.X.shape[0],))
            # self.weights = torch.randn((self.X.shape[0],)) * 0.1

    def resample_dataset(self, split=1.0, seed=None):
        if seed:
            self.set_seed(seed)
        data = [self.X, self.budgets]
        if self.Y is not None:
            data.append(self.Y)
        if self.curves is not None:
            data.append(self.curves)
        if self.weights is not None:
            data.append(self.weights)

        n_samples = int(split * self.X.shape[0])
        n_samples = max(1, n_samples)
        boot_data = resample(*data, n_samples=n_samples)

        self.X = boot_data[0]
        self.budgets = boot_data[1]
        index = 2
        if self.Y is not None:
            self.Y = boot_data[index]
            index += 1
        if self.curves is not None:
            self.curves = boot_data[index]
            index += 1
        if self.weights is not None:
            self.weights = boot_data[index]
            index += 1

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        if self.weights is not None:
            return self.X[idx], self.Y[idx], self.budgets[idx], self.curves[idx], self.weights[idx]
        else:
            return self.X[idx], self.Y[idx], self.budgets[idx], self.curves[idx], self.dummy_weight[0]

    def get_subset(self, idx) -> "TabularDataset":
        subset = self[idx]
        subset_dataset = TabularDataset(
            X=subset[0],
            Y=subset[1],
            budgets=subset[2],
            curves=subset[3],
            use_sample_weights=self.use_sample_weights
        )
        return subset_dataset

    def to(self, device):
        self.X.to(device)
        self.budgets.to(device)
        if self.Y is not None:
            self.Y.to(device)
        if self.curves is not None:
            self.curves.to(device)
        if self.weights is not None:
            self.weights.to(device)
        self.dummy_weight.to(device)

    @staticmethod
    def set_seed(seed):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
