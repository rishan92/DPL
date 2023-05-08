import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class InverseClipSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = 0.5
        self.k = 5

    def forward(self, x: Tensor) -> Tensor:
        x = (torch.log(x / (1 - x)) / self.k) + self.offset

        return x
