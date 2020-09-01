import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

num_node_features = 100
num_classes = 2
num_nodes = 678
num_edges = 1500
num_hidden_nodes = 128

x = torch.randn((num_nodes, num_node_features), dtype=torch.float32)
edge_index = torch.randint(low=0, high=num_nodes, size=(2, num_edges), dtype=torch.long)
y = torch.randint(low=0, high=num_classes, size=(num_nodes,), dtype=torch.long)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_node_features, num_hidden_nodes)
        self.conv2 = GCNConv(num_hidden_nodes, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

data = Data(x=x, edge_index=edge_index, y=y)

net = Net()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
for i in range(1000):
    output = net(data)
    loss = F.cross_entropy(output, data.y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print('Accuracy: ', (torch.argmax(output, dim=1)==data.y).float().mean())