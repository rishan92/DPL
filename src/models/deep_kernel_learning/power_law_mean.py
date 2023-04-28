import gpytorch


class PowerLawMean(gpytorch.means.Mean):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        last_column = x[:, -1]
        return last_column
