import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms

import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim):
        super(Encoder, self).__init__()

        self.c1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.linear = nn.Linear(64 * 4 * 4, hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.var = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        hidden = F.relu(self.c1(x))
        hidden = F.relu(self.c2(hidden))
        hidden = F.relu(self.c3(hidden))
        hidden  = hidden.reshape(-1, 64 * 4 * 4)
        hidden = F.relu(self.linear(hidden))
        z_mu = self.mu(hidden)
        z_var = self.var(hidden)

        return z_mu, z_var

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()

        self.linear1 = nn.Linear(z_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 64 * 4 * 4)
        self.c1 = nn.ConvTranspose2d(64, 32, 3, stride=2)
        self.c2 = nn.ConvTranspose2d(32, 32, 3, stride=2)
        self.c3 = nn.ConvTranspose2d(32, 3, 3, stride=2)

    def forward(self, x):
        hidden = F.relu(self.linear1(x))
        hidden = F.relu(self.linear2(hidden)).reshape(-1, 64, 4, 4)
        hidden = F.relu(self.c1(hidden))
        hidden = F.relu(self.c2(hidden))
        predicted = torch.sigmoid(self.c3(hidden))[:, :, :32, :32]

        return predicted

class BetaVAE(nn.Module):
    def __init__(self, input_dim):
        super(BetaVAE, self).__init__()

        self.hidden_dim = 256
        self.z_dim = 32

        self.beta = 1.
        self.lr = 5e-4

        self.batch_size = 100
        self.n_epochs = 30

        self.enc = Encoder(input_dim, self.hidden_dim, self.z_dim)
        self.enc.to(device)
        self.dec = Decoder(self.z_dim, self.hidden_dim, input_dim)
        self.dec.to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        z_mu, z_var = self.enc(x)

        # Sample
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        z_sample = eps.mul(std).add_(z_mu)

        predicted = self.dec(z_sample)

        return predicted, z_sample, z_mu, z_var

    def loss(self, x, x_reconstructed, z_sample, z_mu, z_var):
        recon_loss = F.mse_loss(x_reconstructed, x, size_average=False)

        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu**2 - 1.0 - z_var)

        return recon_loss + self.beta * kl_loss

    def train_model(self, train_dataset, save_name):
        self.train()

        dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        for i in range(self.n_epochs):
            epoch_loss = 0
            for x in tqdm(dataloader):
                x = x.to(device)

                self.optimizer.zero_grad()

                x_reconstructed, z_sample, z_mu, z_var = self.forward(x)
                loss = self.loss(x, x_reconstructed, z_sample, z_mu, z_var)
                loss.backward()

                epoch_loss += loss.item()

                self.optimizer.step()

            print("Epoch", i, "out of", self.n_epochs, "Avg Loss:", epoch_loss / len(train_dataset))

        torch.save(self, save_name + ".pth")
