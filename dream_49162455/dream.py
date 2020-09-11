import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

from PIL import Image
import numpy as np
from torchvision.transforms import transforms

inv_im_trans = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)

to_tensor = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ), 
])

def save_im(tensor, name):
    im = inv_im_trans(tensor[0].detach().cpu()).numpy().transpose(1, 2, 0)
    im = (im*255).astype(np.uint8)
    Image.fromarray(im).save(name)

cloud = to_tensor(Image.open('cloud.jpg')).unsqueeze(0)
dog = to_tensor(Image.open('dog.jpg')).unsqueeze(0)

net = models.googlenet(pretrained=True).eval()

print(torch.argmax(net(cloud))) # 984 - geyser, well cloud is not one of the classes
print(torch.argmax(net(dog))) # 208 - Labrador retriever

# We start optimization here
dog_class_prob = net(dog).detach()
original_cloud = cloud.clone()
cloud.requires_grad = True
optim = optim.RMSprop(params=[cloud], lr=1e-3)
criterion = nn.MSELoss()

os.makedirs('output', exist_ok=True)
for i in range(50):
    cloud_class_prob = net(cloud)
    optim.zero_grad()
    loss = criterion(cloud_class_prob, dog_class_prob) + criterion(cloud, original_cloud)*100
    print(loss, torch.argmax(cloud_class_prob))
    loss.backward()
    optim.step()

    save_im(cloud, 'output/%d.jpg' % i)
