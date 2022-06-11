import torch
import torch.nn as nn

img = torch.randn((2, 3, 64, 64))
msk = torch.randn((2, 1, 64, 64))

cat_conv = nn.Conv2d(4, 64, kernel_size=3, padding=1, bias=False)
add_conv_i = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
add_conv_m = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)

add_conv_i.weight.data = cat_conv.weight.data[:, :3]
add_conv_m.weight.data = cat_conv.weight.data[:, 3:]

cat_out = cat_conv(torch.cat([img, msk], 1))
add_out = add_conv_i(img) + add_conv_m(msk)

print(torch.allclose(cat_out, add_out, atol=1e-6))
