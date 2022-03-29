import math
import torch
import matplotlib.pyplot as plt

x = torch.linspace(-math.pi, math.pi, 5000, requires_grad=True)
y = torch.sin(x)
y.backward(torch.ones_like(x))
plt.plot(x.detach().numpy(), y.detach().numpy(), label='sin(x)')
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label='cos(x)') # print derivative of sin(x)
plt.show()