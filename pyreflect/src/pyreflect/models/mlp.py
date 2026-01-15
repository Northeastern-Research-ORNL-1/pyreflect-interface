
import numpy as np
import torch
from torch import nn
from ..config.runtime import DEVICE

# Defining a Multilayer Perceptron, MLP.
class MLP(nn.Module):

    def __init__(self, latent_size, n_label):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(latent_size, 36),
            nn.ReLU(),
            nn.Linear(36, 12),
            nn.ReLU(),
            nn.Linear(12, n_label),
        )

    def forward(self, x):
        return self.mlp(x)


class linear_perc(nn.Module):

    def __init__(self, latent_size, n_label):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(latent_size, n_label),
        )

    def forward(self, x):
        return self.mlp(x)


def train(model, train_loader, val_loader, epochs, loss_fn):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=0.0001)
    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        shot_loss = []
        for data, label in train_loader:
            input = data.view(data.size(0), -1).to(DEVICE)
            label = label.to(DEVICE)
            opt.zero_grad()
            output = model(input)
            loss = loss_fn(output, label)
            loss.backward()
            opt.step()
            shot_loss.append(loss.item())
        epoch_loss = np.mean(shot_loss)
        train_loss.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            val_shot_loss = []
            for data, label in val_loader:
                input = data.view(data.size(0), -1).to(DEVICE)
                label = label.to(DEVICE)
                output = model(input)
                loss = loss_fn(output, label)
                val_shot_loss.append(loss.item())
            val_epoch_loss = np.mean(val_shot_loss)
            val_loss.append(val_epoch_loss)

        print('Epoch: ' + str(epoch + 1) + ', train loss: ' + str(epoch_loss) + ', valid loss: ' + str(val_epoch_loss))

    return train_loss, val_loss


class deep_MLP(nn.Module):

    def __init__(self, input_size, n_label):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 80),
            nn.ReLU(),
            nn.Linear(80, n_label),
        )

    def forward(self, x):
        return self.mlp(x)


class interc_MLP(nn.Module):

    def __init__(self, input_size, n_label):
        super().__init__()

        self.mlp1 = nn.Sequential(
            nn.Linear(input_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 80),
            nn.ReLU(),
            nn.Linear(80, 30),
        )

        self.mlp2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(30, n_label),
        )

    def forward(self, x):
        z = self.mlp1(x)
        return self.mlp2(z)

    def interc(self, x):
        return self.mlp1(x)