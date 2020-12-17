import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

torch.manual_seed(0)

def show_tensor_images(image_tensor,num_images=25,size=(1,28,28)):
    image_unflat=image_tensor.detach().cpu().view(-1,*size)
    image_grid=make_grid(image_unflat[:num_images],nrow=5)
    plt.imshow(image_grid.permute(1,2,0).squeeze())
    plt.show()

class Generator(nn.Module):
    def __init__(self,z_dim):
        super(Generator,self).__init__()
        self.linear1=nn.Linear(z_dim,128)
        self.bn1=nn.BatchNorm1d(128)
        self.linear2=nn.Linear(128,256)
        self.bn2=nn.BatchNorm1d(256)
        self.linear3=nn.Linear(256,512)
        self.bn3=nn.BatchNorm1d(512)
        self.linear4=nn.Linear(512,1024)
        self.bn4=nn.BatchNorm1d(1024)
        self.linear5=nn.Linear(1024,784)
        self.relu=nn.ReLU(True)
        self.sigmoid=nn.Sigmoid()
    def forward(self,x):
        x=self.linear1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.linear2(x)
        x=self.bn2(x)
        x=self.relu(x)
        x=self.linear3(x)
        x=self.bn3(x)
        x=self.relu(x)
        x=self.linear4(x)
        x=self.bn4(x)
        x=self.relu(x)
        x=self.linear5(x)
        x=self.sigmoid(x)
        return(x)

def get_noise(n_samples,z_dim,device='cpu'):
    return torch.randn(n_samples,z_dim,device=device)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.linear1=nn.Linear(784,512)
        self.linear2=nn.Linear(512,256)
        self.linear3=nn.Linear(256,128)
        self.linear4=nn.Linear(128,1)
        self.relu=nn.LeakyReLU(0.2,True)
    def forward(self,x):
        x=self.linear1(x)
        x=self.relu(x)
        x=self.linear2(x)
        x=self.relu(x)
        x=self.linear3(x)
        x=self.relu(x)
        x=self.linear4(x)
        return(x)

criterion=nn.BCEWithLogitsLoss()
epochs=200
z_dim=64
display_step=40000
batch_size=128
lr=0.00001
device='cuda'

dataloader=DataLoader(MNIST('.',download=True,transform=transforms.ToTensor()),batch_size=batch_size,shuffle=True)

gen=Generator(z_dim).to(device)
gen_opt=torch.optim.Adam(gen.parameters(),lr=lr)
disc=Discriminator().to(device)
disc_opt=torch.optim.Adam(disc.parameters(),lr=lr)

def get_disc_loss(gen,disc,criterion,real,num_images,z_dim,device):
    noise=get_noise(num_images,z_dim,device=device)
    gen_out=gen(noise)
    disc_fake_out=disc(gen_out.detach())
    fake_loss=criterion(disc_fake_out,torch.zeros_like(disc_fake_out))
    disc_real_out=disc(real)
    real_loss=criterion(disc_real_out,torch.ones_like(disc_real_out))
    disc_loss=(fake_loss+real_loss)/2
    return(disc_loss)

def get_gen_loss(gen,disc,criterion,num_images,z_dim,device):
    noise=get_noise(num_images,z_dim,device=device)
    gen_out=gen(noise)
    disc_out=disc(gen_out)
    loss=criterion(disc_out,torch.ones_like(disc_out))
    return loss

cur_step=0
mean_generator_loss=0
mean_discriminator_loss=0
gen_loss=False
error=False
for epoch in range(epochs):
    for x,_ in tqdm(dataloader):
        cur_batch_size=len(x)
        x=x.view(cur_batch_size,-1).to(device)

        disc_opt.zero_grad()
        disc_loss=get_disc_loss(gen,disc,criterion,x,cur_batch_size,z_dim,device)
        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        gen_opt.zero_grad()
        gen_loss=get_gen_loss(gen,disc,criterion,cur_batch_size,z_dim,device)
        gen_loss.backward()
        gen_opt.step()

        mean_discriminator_loss+=disc_loss.item()/display_step
        mean_generator_loss+=gen_loss.item()/display_step

        if cur_step%display_step==0 and cur_batch_size>0:
            print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = gen(fake_noise)
            show_tensor_images(fake)
            show_tensor_images(x)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        cur_step += 1