import torch
import math
import torch.nn as nn
from typing import Tuple

c = 25
x = torch.ones(1,112,112,128)*c

layer = nn.LayerNorm(normalized_shape=128)
y = layer(x)

print(y)

def layer_norm(
    x: torch.Tensor, dim: Tuple[int], eps: float = 1e-5
) -> torch.Tensor:
    mean = torch.mean(x, dim=dim, keepdim=True)
    var = x.var(dim = dim, keepdim = True, unbiased=False)
    return (x-mean+math.sqrt(eps))/ torch.sqrt(var + eps)

y2 = layer_norm(x,dim=3)
print(y2.max())
