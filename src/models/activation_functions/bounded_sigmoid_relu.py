import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class BoundedSReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        clipped_neg_1 = F.relu(x + 1)
        clipped_1 = 1 - F.relu(2 - clipped_neg_1)

        return clipped_1
