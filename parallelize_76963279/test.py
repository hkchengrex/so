import torch
import torch.nn.functional as F
from time import time

#############################################
# Parameters
#############################################

B = 1
C = 3
H = 4
W = 4
S = 1

src = torch.ones((B, C, H, W), dtype=torch.float32)


indices = torch.zeros((C, H, W), dtype=torch.long)
for i in range(C):
    for y in range(H):
        for x in range(W):
            indices[i, y, x] = x + i*S
indices = indices.view(1, C, H, W).expand(B, C, H, W)

target = torch.zeros((B, C, H, W+(C-1)*S), dtype=torch.float32)
target.scatter_(3, indices, src)
print(target.sum(dim=1, keepdim=True))


# test 2
out_W = W + (C-1)*S
target = torch.zeros((B, H*out_W), dtype=torch.float32)
src = src.view(B, C*H*W)
indices = torch.zeros((C*H*W), dtype=torch.long)
# for i in range(C):
#     for y in range(H):
#         for x in range(W):
#             indices[i*H*W+y*W+x] = x + i*S + y*(out_W)
i_list = torch.arange(C, dtype=torch.long)
y_list = torch.arange(H, dtype=torch.long)
x_list = torch.arange(W, dtype=torch.long)
indices = x_list + i_list.view(C, 1, 1)*S + y_list.view(1, H, 1)*(out_W)


indices = indices.view(1, C*H*W).expand(B, C*H*W)
target.scatter_add_(1, indices, src)

print(target.view(B, 1, H, out_W))
