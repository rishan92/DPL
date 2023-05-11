import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(self, X, budgets, curves=None, Y=None, use_sample_weights=False):
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
        self.dummy_weight = torch.tensor(1)
        if use_sample_weights:
            self.weights = torch.rand((self.X.shape[0],))
            # self.weights = torch.randn((self.X.shape[0],)) * 0.1
            
    def reset_sample_weights(self):
        if self.use_sample_weights:
            self.weights = torch.rand((self.X.shape[0],))

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        if self.use_sample_weights:
            return self.X[idx], self.Y[idx], self.budgets[idx], self.curves[idx], self.weights[idx]
        else:
            return self.X[idx], self.Y[idx], self.budgets[idx], self.curves[idx], self.dummy_weight

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
