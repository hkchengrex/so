import torch
import torch.nn as nn

class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(8)

    def forward(self,x):
        print(x)
        x = self.avgpool(x)
        print(x)
        return x

def test():
    x = torch.Tensor([[[1,2,3,4,5,6]]])
    net = TestNet()
    y = net(x)
    return y

test()