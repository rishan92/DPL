import torch
import torch.nn as nn
from torch import Tensor


class OffsetTanh(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x = torch.tanh(x)
        x = (x + 1) / 2
        return x
