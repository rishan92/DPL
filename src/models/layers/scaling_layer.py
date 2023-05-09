import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Any, Optional


class ScalingLayer(nn.Module):
    def __init__(self, in_features: int, bias: bool = True, device=None, dtype=None, bias_values: List = None):
        super().__init__()
        self.linear = nn.Linear(in_features, in_features, bias, device, dtype)
        self.mask = torch.eye(in_features, dtype=torch.bool)
        if bias_values is not None:
            self.linear.bias = nn.Parameter(torch.tensor(bias_values, device=device))

    def forward(self, x):
        self.linear.weight.data *= self.mask
        x = self.linear(x)
        return x
