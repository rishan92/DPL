import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class InverseClipLeakyReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.negative_slope: float = 1 / 0.01

    def forward(self, x: Tensor) -> Tensor:
        clipped_0 = F.leaky_relu(x, negative_slope=self.negative_slope)

        # Subtract from 1 and apply ReLU to clip values greater than 1 to 1
        clipped_1 = 1 - F.leaky_relu(1 - clipped_0, negative_slope=self.negative_slope)

        return clipped_1
