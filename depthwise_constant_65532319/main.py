import torch
import torch.nn as nn

# Normal conv
normal_conv = nn.Conv2d(1, 2, kernel_size=1)

# We can artificially repeat the weight along the channel dimension -> constant depthwise
repeated_conv = nn.Conv2d(6, 2, kernel_size=1)
repeated_conv.weight.data = normal_conv.weight.data.expand(-1, 6, -1, -1)
repeated_conv.bias.data = normal_conv.bias.data

# data = torch.randn(1, 6, 3, 3)

# # same result
# print(repeated_conv(data))
# print(normal_conv(data.sum(1, keepdim=True)))

data1 = torch.randn(1, 6, 3, 3)
data2 = data1.clone()
data1.requires_grad = True
data2.requires_grad = True

# same result
repeated_conv(data1).mean().backward()
normal_conv(data2.sum(1, keepdim=True)).mean().backward()

print(data1.grad, repeated_conv.weight.grad.sum(1))
print(data2.grad, normal_conv.weight.grad)
