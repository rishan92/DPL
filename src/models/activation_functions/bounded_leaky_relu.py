import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class BoundedLeakyReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        # clipped_neg_1 = F.leaky_relu(x + 1)
        # clipped_1 = 1 - F.leaky_relu(2 - clipped_neg_1)

        clipped_neg_1 = F.leaky_relu(x + 1)
        clipped_1 = (1 - F.leaky_relu(2 - clipped_neg_1))
        clipped = (clipped_1 + 1) / 2

        return clipped
