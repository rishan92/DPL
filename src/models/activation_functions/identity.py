import torch
import torch.nn as nn
from torch import Tensor


class Identity(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x
