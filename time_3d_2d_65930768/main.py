import torch
import torch.nn as nn
import time

b = 4
c = 128
t = 128
h = 16
w = 16
raw_data = torch.randn(b, c, t, h, w).cuda()

def time2D():
        conv2d = nn.Conv2d(c, c, kernel_size=3, padding=1).cuda()
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            data = raw_data.transpose(1,2).reshape(b*t, c, h, w).detach()
            out = conv2d(data)
            out = out.view(b, t, c, h, w).transpose(1, 2)
            out.mean().backward()
        torch.cuda.synchronize()
        end = time.time()
        print(out.shape)
        print("  --- %s ---  " %(end - start))

def time3D():
        conv3d = nn.Conv3d(c, c, kernel_size=(1,3,3), padding=(0,1,1)).cuda()
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            out = conv3d(raw_data.detach())
            out.mean().backward()
        torch.cuda.synchronize()
        end = time.time()
        print(out.shape)
        print("  --- %s ---  " %(end - start))

print("Initializing cuda state")
time2D()

print("going to time2D")
time2D()
print("going to time3D")
time3D()