import torch
import torch.nn.functional as F
from time import time

#############################################
# Parameters
#############################################

B = 16
C = 28
H = 256
W = 256
S = 2
T = 1000
device = torch.device('cuda')

seed = 2023
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#############################################
# Method 1
#############################################

alpha = torch.zeros(B, 1, 1, W+(C-1)*S, device=device)
for i in range(C):
    alpha[..., (i*S):(i*S+W)] += 1

def A(x, mask):
    z = x * mask
    y = torch.zeros(B, 1, H, W+(C-1)*S, device=x.device)
    for i in range(C):
        y[..., (i*S):(i*S+W)] += z[:, (i):(i+1)]
    return y

def A_pinv(y, mask):
    z = y / alpha.to(y.device)
    x = torch.cat([z[..., (i*S):(i*S+W)] for i in range(C)], dim=1) / mask
    return x

#############################################
# Method 2
#############################################

kernel = torch.zeros(1, C, 1, (C-1)*S+1, device=device)
for i in range(C):
    kernel[:, C-i-1, :, i*S] = 1

def A_fast(x, mask):
    return F.conv2d(x * mask, kernel.to(x.device), padding=(0, (C-1)*S))

def A_pinv_fast(y, mask):
    return F.conv_transpose2d(y / alpha.to(y.device), kernel, padding=(0, (C-1)*S)) / mask


#############################################
# Method 3
#############################################
out_W = W + (C-1)*S
i_list = torch.arange(C, dtype=torch.long, device=device)
y_list = torch.arange(H, dtype=torch.long, device=device)
x_list = torch.arange(W, dtype=torch.long, device=device)
indices = x_list + i_list.view(C, 1, 1)*S + y_list.view(1, H, 1)*(out_W)
indices = indices.view(1, C*H*W).expand(B, C*H*W)
"""
functionally equivalent to:
for i in range(C):
    for y in range(H):
        for x in range(W):
            indices[i*H*W+y*W+x] = x + i*S + y*(out_W)
"""

def A_faster(x, mask):
    y = torch.zeros(B, H*out_W, device=x.device)
    y.scatter_add_(1, indices, (x*mask).view(B, C*H*W))
    return y.view(B, 1, H, out_W)


#############################################
# Test 1
#############################################
torch.cuda.synchronize()
start_time = time()
for i in range(T):
    x = torch.rand(B, C, H, W, device=device)
    mask = torch.rand(1, 1, H, W, device=device)
    mask[mask == 0] = 1e-12
    y = A(x, mask)
torch.cuda.synchronize()
end_time = time()
print('---')
print('Test 1')
print('Running Time:', end_time - start_time)

#############################################
# Test 2
#############################################
torch.cuda.synchronize()
start_time = time()
for i in range(T):
    x = torch.rand(B, C, H, W, device=device)
    mask = torch.rand(1, 1, H, W, device=device)
    mask[mask == 0] = 1e-12
    y = A_fast(x, mask)
torch.cuda.synchronize()
end_time = time()
print('---')
print('Test 2')
print('Running Time:', end_time - start_time)

#############################################
# Test 3
#############################################
torch.cuda.synchronize()
start_time = time()
for i in range(T):
    x = torch.rand(B, C, H, W, device=device)
    mask = torch.rand(1, 1, H, W, device=device)
    mask[mask == 0] = 1e-12
    y = A_faster(x, mask)
torch.cuda.synchronize()
end_time = time()
print('---')
print('Test 3')
print('Running Time:', end_time - start_time)


error = 0
for _ in range(T):
    error += (A(x, mask) - A_fast(x, mask)).abs().mean()
error /= T
print('---')
print('|Test1 - Test2|: ', error)

error = 0
for _ in range(T):
    error += (A(x, mask) - A_faster(x, mask)).abs().mean()
error /= T
print('---')
print('|Test1 - Test3|: ', error)

error = 0
for _ in range(T):
    error += (A_fast(x, mask) - A_faster(x, mask)).abs().mean()
error /= T
print('---')
print('|Test2 - Test3|: ', error)