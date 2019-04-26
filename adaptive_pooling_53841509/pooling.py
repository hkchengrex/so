import math
import torch
import torch.nn as nn

in_length = 5
out_length = 3

x = torch.arange(0, in_length).view(1, 1, -1).float()
print(x)

stride = (in_length//out_length)
avg_pool = nn.AvgPool1d(
        stride=stride,
        kernel_size=(in_length-(out_length-1)*stride),
        padding=0,
    )
adaptive_pool = nn.AdaptiveAvgPool1d(out_length)

print(avg_pool.stride, avg_pool.kernel_size)

y_avg = avg_pool(x)
y_ada = adaptive_pool(x)

print(y_avg)
print(y_ada)
print('Error: ', (y_avg - y_ada).abs().sum().item())
