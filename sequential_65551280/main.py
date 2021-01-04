import torch
import torch.nn as nn

seed = 0
layernum = 3
torch.manual_seed(seed)
# ====== net_a =====
def get_net():
    layers = [ nn.Linear(7, 64), nn.Tanh()]
    for i in range(layernum-1): # layernum = 3
        layers.append(nn.Linear(64, 64))
        layers.append(nn.Tanh())
    layers.append(nn.Linear(64, 8))
    return layers

net_x = nn.Sequential(*get_net())
net_y = nn.Sequential(*get_net())
net_z = nn.Sequential(*get_net())


data = torch.randn(8, 7)

print(net_x[0].weight.data[0])

net_x(data).mean().backward()

print(net_x[0].weight.grad.data[0])
print(net_y[0].weight.grad.data[0])
