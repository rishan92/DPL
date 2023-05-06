import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class ClipReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        clipped_0 = F.relu(x)

        # Subtract from 1 and apply ReLU to clip values greater than 1 to 1
        clipped_1 = 1 - F.relu(1 - clipped_0)

        return clipped_1
