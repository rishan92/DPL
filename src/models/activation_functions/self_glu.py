import torch
import torch.nn as nn
from torch import Tensor


class SelfGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.act_fn = nn.GLU()

    def forward(self, x: Tensor) -> Tensor:
        x = torch.cat((x, x))
        x = self.act_fn(x)
        return x

