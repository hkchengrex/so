import torch
import torch.nn as nn
import numpy as np

predictions = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)

def l1_loss_smooth(predictions, targets, beta = 1.0):
    
    loss = 0

    diff = predictions-targets
    mask = (diff.abs() < beta)
    loss += mask * (0.5*diff**2 / beta)
    loss += (~mask) * (diff.abs() - 0.5*beta)
    
    return loss.mean()

output = l1_loss_smooth(predictions, target)
print(output)

loss = nn.SmoothL1Loss(beta=1.0)
output = loss(predictions, target)
print(output)
