import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class InverseOffsetTanh(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x = 2 * x - 1
        x = torch.atanh(x)
        return x
