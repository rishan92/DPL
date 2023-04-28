import torch
import torch.nn as nn
from torch import Tensor


class Abs(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x = torch.abs(x)
        return x

