import numpy as np
import torch
import torch.utils
import torch.nn as nn
import torch.nn.functional as F

# set processing device
from ..config.runtime import DEVICE as device

# Encoder class, takes in input data dimension and desired latent space size
class Encoder(nn.Module):
    def __init__(self, init_size, lat_size):
        super(Encoder, self).__init__()

        # Create layers of neurons and activation fucntions
        self.layers = nn.Sequential(
            nn.Linear(init_size, 500),
            nn.ReLU(),
            nn.Linear(500, 300),
            nn.ReLU(),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, 72),
            nn.ReLU(),
            nn.Linear(72, lat_size),
        )

    # Feed input through layers
    def forward(self, x):
        z = self.layers(x)
        return z


# Encoder class for VAE, also takes in input data dimension and latent space size
class VariationalEncoder(nn.Module):
    def __init__(self, init_size, lat_size):
        super(VariationalEncoder, self).__init__()

        # Creating layers of neurons and activation functions for VAE
        self.layers = nn.Sequential(
            nn.Linear(init_size, 500),
            nn.ReLU(),
            nn.Linear(500, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU()
        )

        # Creating last layer of VAE encoder, with two seperate neuron sets
        self.linear2 = nn.Linear(100, lat_size)
        self.linear3 = nn.Linear(100, lat_size)

        # Creating Kullback-Liebler (kl) divergence
        self.kl = 0

    # Feeding input into layers
    def forward(self, x):
        z = self.layers(x)
        # Generating both a mean (mu) and std deviation (sigma) from input
        mu = self.linear2(z)
        sigma = self.linear3(z)
        std = torch.exp(0.5 * sigma)
        # Noise to create vector probabilitcally within distribution created above
        noise = torch.randn_like(std)
        # Calculating KL divergence
        self.kl = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        return mu + (std * noise)


# Decoder class, inverse of encoder layer, decodes latent vectors into original-like data
class Decoder(nn.Module):
    def __init__(self, init_size, lat_size):
        super(Decoder, self).__init__()

        # Create layers of neurons and activation functions
        self.layers = nn.Sequential(
            nn.Linear(lat_size, 72),
            nn.ReLU(),
            nn.Linear(72, 200),
            nn.ReLU(),
            nn.Linear(200, 300),
            nn.ReLU(),
            nn.Linear(300, 500),
            nn.ReLU(),
            nn.Linear(500, init_size),
        )

    # Feed input through layers
    def forward(self, z):
        x = self.layers(z)
        return torch.sigmoid(x)


# Main autoencoder model class, reduces input data down via encoder and then recreates it via decoder
class Autoencoder(nn.Module):
    def __init__(self, init_size, latent_size):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(init_size, latent_size)
        self.decoder = Decoder(init_size, latent_size)

    # Feed input through encoder and decoder
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# VAE model class, works same as above AE but uses probability vectors in latent space
class VariationalAutoencoder(nn.Module):
    def __init__(self, init_size, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(init_size, latent_dims)
        self.decoder = Decoder(init_size, latent_dims)

    # Feed input through encoder and decoder
    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        return self.decoder(z)

# Regular AE training
def train(model, train_loader, val_loader, epochs, loss_fn):
    # set model to training mode, set optimizer as AdamW
    model.train()
    opt = torch.optim.Adam(model.parameters())
    #
    train_loss = []
    val_loss = []

    for epoch in range(epochs):
        shot_loss = []
        for data in train_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            opt.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(inputs, outputs)
            loss.backward()
            opt.step()
            shot_loss.append(loss.item())
        epoch_loss = np.mean(shot_loss)
        train_loss.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            val_shot_loss = []
            for data in val_loader:
                inputs, _ = data
                inputs = inputs.to(device)
                outputs = model(inputs)
                loss = loss_fn(inputs, outputs)
                val_shot_loss.append(loss.item())
            val_epoch_loss = np.mean(val_shot_loss)
            val_loss.append(val_epoch_loss)

        print('Epoch: ' + str(epoch + 1) + ', train loss: ' + str(epoch_loss) + ', valid loss: ' + str(val_epoch_loss))

    return train_loss, val_loss


## VAE train method
# takes in model (VAE), training dataloader, and validation dataloader, number of epochs to train, and the KL weight (beta)
def train_vae(model, train_loader, val_loader, epochs, beta):
    model.train()
    opt = torch.optim.Adam(model.parameters())
    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        shot_loss = []
        for data in train_loader:
            inputs, _ = data
            inputs = inputs.to(device)
            opt.zero_grad()
            outputs = model(inputs)
            ##main_loss = loss_fn(inputs,outputs)
            main_loss = F.binary_cross_entropy(outputs, inputs, reduction='sum')
            loss = main_loss + (beta * model.encoder.kl)
            loss.backward()
            opt.step()
            shot_loss.append(loss.item())
        epoch_loss = np.mean(shot_loss)
        train_loss.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            val_shot_loss = []
            for data in val_loader:
                inputs, _ = data
                inputs = inputs.to(device)
                outputs = model(inputs)
                ##main_loss = loss_fn(inputs,outputs)
                main_loss = F.binary_cross_entropy(outputs, inputs, reduction='sum')
                loss = main_loss + (beta * model.encoder.kl)
                val_shot_loss.append(loss.item())
            val_epoch_loss = np.mean(val_shot_loss)
            val_loss.append(val_epoch_loss)

        print('Epoch: ' + str(epoch + 1) + ', train loss: ' + str(epoch_loss) + ', valid loss: ' + str(val_epoch_loss))

    return train_loss, val_loss
