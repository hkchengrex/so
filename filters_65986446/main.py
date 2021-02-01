import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt

torch.random.manual_seed(42)

transform = transforms.Compose([transforms.ToTensor()])
img = transform(Image.open('dog.jpg')).unsqueeze(0)
print('Image size: ', img.shape)

filters = torch.randn(64, 3, 7, 7)

out = F.conv2d(img, filters)
print('Output size: ', out.shape)

list_of_images = [out[:,i] for i in range(64)]

grid = torchvision.utils.make_grid(list_of_images, normalize=True)
plt.imshow(grid.numpy().transpose(1,2,0))
plt.show()
