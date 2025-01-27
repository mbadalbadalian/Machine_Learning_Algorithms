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

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.fc1 = nn.Linear(latent_size, 400)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(400, 784)
        self.sigmoid = nn.Sigmoid()
        return

    def forward(self, z):
        z = self.relu(self.fc1(z))
        output = self.sigmoid(self.fc2(z))
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(400, 1)
        self.sigmoid = nn.Sigmoid()
        return

    def forward(self,x):
        x = x.view(-1,784)
        x = self.relu(self.fc1(x))
        output = self.sigmoid(self.fc2(x))
        return output

def train(generator, generator_optimizer, discriminator, discriminator_optimizer):
    generator.train()
    discriminator.train()
    avg_generator_loss = 0
    avg_discriminator_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        real_labels = torch.ones(data.size(0), 1).to(device)
        fake_labels = torch.zeros(data.size(0), 1).to(device)

        # Train Discriminator
        discriminator_optimizer.zero_grad()
        real_output = discriminator(data)
        real_loss = nn.BCELoss()(real_output, real_labels)
        
        z = torch.randn(data.size(0), latent_size).to(device)
        fake_data = generator(z)
        fake_output = discriminator(fake_data.detach())
        fake_loss = nn.BCELoss()(fake_output, fake_labels)

        discriminator_loss = real_loss + fake_loss
        discriminator_loss.backward()
        discriminator_optimizer.step()
        avg_discriminator_loss += discriminator_loss.item()

        # Train Generator
        generator_optimizer.zero_grad()
        fake_output = discriminator(fake_data)
        generator_loss = nn.BCELoss()(fake_output, real_labels)
        generator_loss.backward()
        generator_optimizer.step()
        avg_generator_loss += generator_loss.item()

    avg_generator_loss /= len(train_loader)
    avg_discriminator_loss /= len(train_loader)

    return avg_generator_loss, avg_discriminator_loss

def test(generator, discriminator):
    generator.eval()
    discriminator.eval()
    avg_generator_loss = 0
    avg_discriminator_loss = 0

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            real_labels = torch.ones(data.size(0), 1).to(device)
            fake_labels = torch.zeros(data.size(0), 1).to(device)

            # Test Discriminator
            real_output = discriminator(data)
            real_loss = nn.BCELoss()(real_output, real_labels)

            z = torch.randn(data.size(0), latent_size).to(device)
            fake_data = generator(z)
            fake_output = discriminator(fake_data)
            fake_loss = nn.BCELoss()(fake_output, fake_labels)

            discriminator_loss = real_loss + fake_loss
            avg_discriminator_loss += discriminator_loss.item()

            # Test Generator
            fake_output = discriminator(fake_data)
            generator_loss = nn.BCELoss()(fake_output, real_labels)
            avg_generator_loss += generator_loss.item()

    avg_generator_loss /= len(test_loader)
    avg_discriminator_loss /= len(test_loader)

    return avg_generator_loss, avg_discriminator_loss

#################################################################################################################
### Main Functions

#The main function is the driver for the code
if __name__ == "__main__":
        
    #Variables
    if not os.path.exists('results'):
        os.mkdir('results')

    batch_size = 100
    latent_size = 20
    
    epochs = 50

    discriminator_avg_train_losses = []
    discriminator_avg_test_losses = []
    generator_avg_train_losses = []
    generator_avg_test_losses = []

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    kwargs = {'num_workers':1,'pin_memory':True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data',train=True,download=True,transform=transforms.ToTensor()),batch_size=batch_size,shuffle=True,**kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data',train=False,transform=transforms.ToTensor()),batch_size=batch_size,shuffle=True,**kwargs)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    generator_optimizer = optim.Adam(generator.parameters(),lr=1e-3)
    discriminator_optimizer = optim.Adam(discriminator.parameters(),lr=1e-3)

    for epoch in range(1,epochs+1):
        generator_avg_train_loss, discriminator_avg_train_loss = train(generator, generator_optimizer, discriminator, discriminator_optimizer)
        generator_avg_test_loss, discriminator_avg_test_loss = test(generator, discriminator)

        discriminator_avg_train_losses.append(discriminator_avg_train_loss)
        generator_avg_train_losses.append(generator_avg_train_loss)
        discriminator_avg_test_losses.append(discriminator_avg_test_loss)
        generator_avg_test_losses.append(generator_avg_test_loss)

        if epoch%10 == 0:
            with torch.no_grad():
                sample = torch.randn(64, latent_size).to(device)
                sample = generator(sample).cpu()
                save_image(sample.view(64,1,28,28),'Results/A4_E1_Q1_b/sample_'+str(epoch)+'_epochs.png')
                print('Epoch #'+str(epoch))
                display(Image('results/sample_'+str(epoch)+'.png'))
                print('\n')

    plt.plot(discriminator_avg_train_losses)
    plt.plot(generator_avg_train_losses)
    plt.title('Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Disc','Gen'], loc='upper right')
    plt.show()

    plt.plot(discriminator_avg_test_losses)
    plt.plot(generator_avg_test_losses)
    plt.title('Test Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Disc','Gen'], loc='upper right')
    plt.show()