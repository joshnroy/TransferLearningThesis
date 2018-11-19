import torch
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
import cv2
from tensorboardX import SummaryWriter

pics_folder = "randpics/"
writer = SummaryWriter("runs/test1")

def read_file(i_episode, j_step):
    filename = pics_folder + "observation_" + str(i_episode) + "_" + \
    str(j_step) + ".jpg"
    if os.path.isfile(filename):
        img = cv2.imread(filename)
        img = img.flatten()
        return img
    else:
        return None

def read_next_file():
    i_episode = 0
    j_step = 0
    ret = read_file(i_episode, j_step)
    while True:
        while ret is not None:
            while ret is not None:
                yield ret
                j_step += 1
                ret = read_file(i_episode, j_step)
            i_episode += 1
            j_step = 0
            ret = read_file(i_episode, j_step)
        i_episode = 0
        j_step = 0
        ret = read_file(i_episode, j_step)


def read_next_batch(batch_size):
    imggen = read_next_file()
    img = next(imggen)
    while True:
        batch = []
        for _ in range(batch_size):
            if img is None:
                return None
            else:
                batch.append(img)
                img = next(imggen)
        yield np.array(batch)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


first_img = cv2.imread(pics_folder + "observation_0_0.jpg", flags=cv2.IMREAD_COLOR)

mb_size = 50
Z_dim = 128
X_dim = first_img.shape[0] * first_img.shape[1] * first_img.shape[2]
h_dim = 128
c = 0
lr = 1e-6

beta = 1.

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
    h = nn.relu(x @ Wxh + bxh.repeat(x.size(0), 1))
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


def main():
    # Training
    params = [Wxh, bxh, Whz_mu, bhz_mu, Whz_var, bhz_var, Wzh, bzh, Whx, bhx]

    solver = optim.Adam(params, lr=lr)
    batchgen = read_next_batch(mb_size)

    for it in range(100000):
        X = next(batchgen)
        X = Variable(torch.from_numpy(X)).to(device)
        X = X.float() / 255.
        assert ((X >= 0.).all() and (X <= 1.).all())

        # Forward
        z_mu, z_var = Q(X)
        z = sample_z(z_mu, z_var)
        X_sample = P(z)

        # Loss
        assert ((X_sample >= 0.).all() and (X_sample <= 1.).all())
        recon_loss = nn.binary_cross_entropy(X_sample, X, size_average=False)
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var)
        loss = recon_loss + beta * kl_loss

        writer.add_scalar("Reconstruction Loss", recon_loss, it)
        writer.add_scalar("KL Loss", kl_loss, it)
        writer.add_scalar("Total Loss", loss, it)


        if it % 100 == 0:
            torch.save(Wxh, "models/test1/Wxh")
            torch.save(bxh, "models/test1/bxh")
            torch.save(Whz_mu, "models/test1/Whz_mu")
            torch.save(bhz_mu, "models/test1/bhz_mu")
            torch.save(Whz_var, "models/test1/Whz_var")
            torch.save(bhz_var, "models/test1/bhz_var")
            torch.save(Wzh, "models/test1/Wzh")
            torch.save(bzh, "models/test1/bzh")
            torch.save(Whx, "models/test1/Whx")
            torch.save(bhx, "models/test1/bhx")
            print("saved iter", it)
        
        if it % 1000 == 0:
            X_sample_numpy = X_sample.detach().cpu().numpy()
            X_sample_numpy = X_sample_numpy.reshape(50, X_dim)
            one_img = X_sample_numpy[1, :].reshape(first_img.shape[0],
                    first_img.shape[1], first_img.shape[2])
            one_img = np.round(one_img * 255.)
            print(one_img.shape, np.max(one_img), np.min(one_img))
            cv2.imwrite("reconstructed1/img" + str(it) + ".jpg", one_img)
            print("wrote image")

        # Backward
        loss.backward()

        # Update
        solver.step()

        # Housekeeping
        for p in params:
            p.grad.data.zero_()

if __name__ == "__main__":
    main()
