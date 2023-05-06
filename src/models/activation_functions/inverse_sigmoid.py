import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class InverseSigmoid(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x = torch.log(x / (1 - x))

        return x
