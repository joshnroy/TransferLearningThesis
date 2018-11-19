import torch
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)
mb_size = 64
Z_dim = 128
X_dim = mnist.train.images.shape[1]
y_dim = mnist.train.labels.shape[1]
h_dim = 128
c = 0
lr = 1e-3

# Prereqs for encoder network
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size, device=device) * xavier_stddev, requires_grad=True)

Wxh = xavier_init(size = [X_dim, h_dim])
bxh = Variable(torch.zeros(Z_dim, device=device), requires_grad=True)

Whz_mu = xavier_init(size = [h_dim, Z_dim])
bhz_mu = Variable(torch.zeros(Z_dim, device=device), requires_grad=True)

Whz_var  = xavier_init(size = [h_dim, Z_dim])
bhz_var = Variable(torch.zeros(Z_dim, device=device), requires_grad=True)

# Encoder Network
def Q(x):
    h = nn.relu(X @ Wxh + bxh.repeat(X.size(0), 1))
    z_mu = h @ Whz_mu + bhz_mu.repeat(h.size(0), 1)
    z_var = h @ Whz_var + bhz_var.repeat(h.size(0), 1)
    return z_mu, z_var

# Sample from encoder network
def sample_z(mu, log_var):
    # Using reparameterization trick to sample from a gaussian
    eps = Variable(torch.randn(mb_size, Z_dim)).to(device)
    return mu + torch.exp(log_var / 2) * eps

# Pre-reqs for decoder network
Wzh = xavier_init(size = [Z_dim, h_dim])
bzh = Variable(torch.zeros(h_dim, device=device), requires_grad=True)

Whx = xavier_init(size = [h_dim, X_dim])
bhx = Variable(torch.zeros(X_dim, device=device), requires_grad=True)

# Decoder Network
def P(z):
    h = nn.relu(z @ Wzh + bzh.repeat(z.size(0), 1))
    X = nn.sigmoid(h @ Whx + bhx.repeat(h.size(0), 1))
    return X


# Training

params = [Wxh, bxh, Whz_mu, bhz_mu, Whz_var, bhz_var, Wzh, bzh, Whx, bhx]

solver = optim.Adam(params, lr=lr)

for it in range(100000):
    X, _ = mnist.train.next_batch(mb_size)
    X = Variable(torch.from_numpy(X)).to(device)

    # Forward
    z_mu, z_var = Q(X)
    z = sample_z(z_mu, z_var)
    X_sample = P(z)

    # Loss
    recon_loss = nn.binary_cross_entropy(X_sample, X, size_average=False)
    kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var)
    loss = recon_loss + kl_loss


    if it % 1000 == 0:
        print(it, loss)

    # Backward
    loss.backward()

    # Update
    solver.step()

    # Housekeeping
    for p in params:
        p.grad.data.zero_()