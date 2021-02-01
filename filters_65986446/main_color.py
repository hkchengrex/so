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

# list_of_filters = [filters[i,:] for i in range(64)]
# grid = torchvision.utils.make_grid(list_of_filters, normalize=True)
# plt.imshow(grid.numpy().transpose(1,2,0))
# plt.show()

color_out = []
for i in range(3):
    color_out.append(F.conv2d(img[:,i:i+1], filters[:,i:i+1]))
out = torch.stack(color_out, 2)
print('Output size: ', out.shape)

list_of_images = [out[0,i] for i in range(64)]
print(list_of_images[0].shape)

grid = torchvision.utils.make_grid(list_of_images, normalize=True)
plt.imshow(grid.numpy().transpose(1,2,0))
plt.show()
