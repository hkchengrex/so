import torch

class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start, end):
        super().__init__()
        self.start = start
        self.end = end
        
    def __iter__(self):
        return iter(range(self.start, self.end))

dataset = MyIterableDataset(0, 4)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=1, drop_last=False)

for epoch in range(2):
    for i, data in enumerate(dataloader):
        print(i, data)