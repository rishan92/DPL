import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class ClipLeakyReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        clipped_0 = F.leaky_relu(x)

        # Subtract from 1 and apply ReLU to clip values greater than 1 to 1
        clipped_1 = 1 - F.leaky_relu(1 - clipped_0)

        return clipped_1
