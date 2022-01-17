import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, RandomSampler


class ToyDataset(Dataset):
    def __init__(self, type):
        self.type = type

    def __getitem__(self, idx):
        return f'{self.type}, {idx}'

    def __len__(self):
        return 10

def get_sampler(dataset, seed=42):
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = RandomSampler(dataset, generator=generator)
    return sampler


original_dataset = ToyDataset('original')
pcp_dataset = ToyDataset('pcp')

original_loader = DataLoader(original_dataset, batch_size=2, sampler=get_sampler(original_dataset))
pcp_loader = DataLoader(pcp_dataset, batch_size=2, sampler=get_sampler(pcp_dataset))

for data in original_loader:
    print(data)

for data in pcp_loader:
    print(data)