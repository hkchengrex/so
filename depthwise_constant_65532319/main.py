import torch
import torch.nn as nn

# Normal conv
normal_conv = nn.Conv2d(1, 2, kernel_size=1)

# We can artificially repeat the weight along the channel dimension -> constant depthwise
repeated_conv = nn.Conv2d(6, 2, kernel_size=1)
repeated_conv.weight.data = normal_conv.weight.data.expand(-1, 6, -1, -1)
repeated_conv.bias.data = normal_conv.bias.data

data = torch.randn(1, 6, 3, 3)

# same result
print(repeated_conv(data))
print(normal_conv(data.sum(1, keepdim=True)))
