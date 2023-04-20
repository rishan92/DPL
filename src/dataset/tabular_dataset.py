from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(self, X, budgets, curves=None, Y=None):
        super().__init__()
        self.X = X
        self.Y = Y
        self.budgets = budgets
        self.curves = curves

    def __len__(self):
        return self.Y.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.budgets[idx], self.curves[idx]

    def to(self, device):
        self.X.to(device)
        self.budgets.to(device)
        if self.Y is not None:
            self.Y.to(device)
        if self.curves is not None:
            self.curves.to(device)
