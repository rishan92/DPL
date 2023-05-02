import torch
import torch.nn as nn
from torch import Tensor


class Clamp(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x = torch.clamp(x, min=0, max=1)
        return x
