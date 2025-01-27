from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from IPython.display import Image, display
import matplotlib.pyplot as plt
import os

#################################################################################################################
### Other Functions

class VAE(nn.Module):
    def __init__(self):
        super(VAE,self).__init__()
        self.fully_connected_layer1 = nn.Linear(784,400)
        self.relu = nn.ReLU()
        self.fc_mean = nn.Linear(400,latent_size)
        self.fc_logvar = nn.Linear(400,latent_size)
        self.fully_connected_layer2 = nn.Linear(latent_size,400)
        self.fully_connected_layer3 = nn.Linear(400,784)
        self.sigmoid = nn.Sigmoid()
        return

    def encode(self, x):
        x = self.relu(self.fully_connected_layer1(x))
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean+eps*std

    def decode(self, z):
        z = self.relu(self.fully_connected_layer2(z))
        reconstruction = self.sigmoid(self.fully_connected_layer3(z))
        return reconstruction

    def forward(self, x):
        x = x.view(-1, 784)
        mean,logvar = self.encode(x)
        z = self.reparameterize(mean,logvar)
        reconstruction = self.decode(z)
        return reconstruction, mean, logvar

def vae_loss_function(reconstructed_x, x, means, log_variances):
    BCE = nn.BCELoss(reduction='sum')(reconstructed_x, x.view(-1, 784))
    KLD = -0.5 * torch.sum(1 + log_variances - means.pow(2) - log_variances.exp())
    loss = BCE + KLD
    reconstruction_loss = BCE
    return loss,reconstruction_loss

def train(model, optimizer):
    model.train()
    train_loss = 0
    train_reconstruction_loss = 0

    for batch_idx,(data,_) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mean, logvar = model(data)
        loss, recon_loss = vae_loss_function(recon_batch, data, mean, logvar)
        loss.backward()
        train_loss += loss.item()
        train_reconstruction_loss += recon_loss.item()
        optimizer.step()

    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_reconstruction_loss = train_reconstruction_loss / len(train_loader.dataset)
    return avg_train_loss, avg_train_reconstruction_loss

def test(model):
    model.eval()
    test_loss = 0
    test_reconstruction_loss = 0

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mean, logvar = model(data)
            loss, recon_loss = vae_loss_function(recon_batch, data, mean, logvar)
            test_loss += loss.item()
            test_reconstruction_loss += recon_loss.item()

    avg_test_loss = test_loss/len(test_loader.dataset)
    avg_test_reconstruction_loss = test_reconstruction_loss/len(test_loader.dataset)
    return avg_test_loss, avg_test_reconstruction_loss

#################################################################################################################
### Main Functions

#The main function is the driver for the code
if __name__ == "__main__":
    
    #Variables
    if not os.path.exists('results'):
        os.mkdir('results')

    batch_size = 100
    latent_size = 20

    #Main Code   
    epochs = 50
    avg_train_losses = []
    avg_train_reconstruction_losses = []
    avg_test_losses = []
    avg_test_reconstruction_losses = []

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data',train=True,download=True,transform=transforms.ToTensor()),batch_size=batch_size, shuffle=True,**kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data',train=False,transform=transforms.ToTensor()),batch_size=batch_size,shuffle=True,**kwargs)

    vae_model = VAE().to(device)
    vae_optimizer = optim.Adam(vae_model.parameters(),lr=1e-3)

    for epoch in range(1,epochs+1):
        avg_train_loss,avg_train_reconstruction_loss = train(vae_model,vae_optimizer)
        avg_test_loss,avg_test_reconstruction_loss = test(vae_model)
        
        avg_train_losses.append(avg_train_loss)
        avg_train_reconstruction_losses.append(avg_train_reconstruction_loss)
        avg_test_losses.append(avg_test_loss)
        avg_test_reconstruction_losses.append(avg_test_reconstruction_loss)

        if epoch%10 == 0:
            with torch.no_grad():
                sample = torch.randn(64,latent_size).to(device)
                sample = vae_model.decode(sample).cpu()
                save_image(sample.view(64,1,28,28),'results/sample_'+str(epoch)+'.png')
                print('Epoch #'+str(epoch))
                display(Image('results/sample_'+str(epoch)+'.png'))
                print('\n')

    plt.plot(avg_train_reconstruction_losses)
    plt.title('Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch #')
    plt.show()

    plt.plot(avg_test_reconstruction_losses)
    plt.title('Test Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch #')
    plt.show()