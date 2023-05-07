import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class ClipSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = 0.5
        self.k = 5

    def forward(self, x: Tensor) -> Tensor:
        x = 1 / (1 + torch.exp(-1 * self.k * (x - self.offset)))

        return x
