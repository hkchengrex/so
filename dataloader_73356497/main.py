import torch
from torch.utils.data import DataLoader

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_size=50000):
        self.data_size = data_size

    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, idx):
        # print(idx)
        return idx

dataset = MyDataset()
dl = DataLoader(dataset)
print(len(dl))

for j, i in enumerate(dl):
  if j%10000 == 0:
    print(j)

dl = DataLoader(dataset, batch_size=2)
print(len(dl))

for j, i in enumerate(dl):
  if j%10000 == 0:
    print(j)