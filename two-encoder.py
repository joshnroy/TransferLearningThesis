import torch
from torch import nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
import cv2
from tensorboardX import SummaryWriter
import sys

import gym
import gym_cartpole_visual

prev_runs = os.listdir("runs/")

test_num = int(max(prev_runs)[4]) + 1

writer = SummaryWriter("runs/test" + str(test_num))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_size = (45, 80)
batch_size = 100
state_size = 5
rp_size = 50
action_size = 1
image_dimension = img_size[0] * img_size[1] * 3
action_dimension = 2

beta = 1e-3
prediction_loss_term = 0.

# Network
class RPencoder(nn.Module):
    def __init__(self):
        super(RPencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 200, (4, 3), stride=(4, 3)), # b, 200, 20, 15
            nn.ReLU(True),
            nn.MaxPool2d((4, 3), stride=(4, 3)), # b, 200, 5, 5
            nn.Conv2d(200, 100, 3, stride=2, padding=1), # b, 100, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1), # b, 8, 2, 2
        )

        self.linear_encoder_mu = nn.Sequential(
            nn.Linear(100 * 2 * 2, rp_size), # b, rp_size
            nn.ReLU(True)
        )

        self.linear_encoder_sigma = nn.Sequential(
            nn.Linear(100 * 2 * 2, rp_size), # b, rp_size
            nn.ReLU(True)
        )

        self.linear_encoder = nn.Sequential(
            nn.Linear(100 * 2 * 2, rp_size), # b, rp_size
            nn.ReLU(True)
        )

    # Sample from encoder network
    def sample_z(self, mu, log_var):
        # Using reparameterization trick to sample from a gaussian
        eps = Variable(torch.randn(batch_size, rp_size)).to(device)
        return mu + torch.exp(log_var / 2) * eps

    def forward (self, x):
        # Encode
        conved = self.encoder(x)
        conved = conved.reshape(100, 100 * 2 * 2)
        mu = self.linear_encoder_mu(conved)
        sigma = self.linear_encoder_sigma(conved)

        render_params = self.linear_encoder(conved)
        # render_params = self.sample_z(mu, sigma)

        return render_params, mu, sigma

class Sencoder(nn.Module):
    def __init__(self):
        super(Sencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 200, (4, 3), stride=(4, 3)), # b, 200, 20, 15
            nn.ReLU(True),
            nn.MaxPool2d((4, 3), stride=(4, 3)), # b, 200, 5, 5
            nn.Conv2d(200, 100, 3, stride=2, padding=1), # b, 100, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1), # b, 8, 2, 2
        )

        self.linear_encoder_mu = nn.Sequential(
            nn.Linear(100 * 2 * 2, state_size), # b, state_size
            nn.ReLU(True)
        )

        self.linear_encoder_sigma = nn.Sequential(
            nn.Linear(100 * 2 * 2, state_size), # b, state_size
            nn.ReLU(True)
        )

        self.linear_encoder = nn.Sequential(
            nn.Linear(100 * 2 * 2, state_size), # b, state_size
            nn.ReLU(True)
        )

    # Sample from encoder network
    def sample_z(self, mu, log_var):
        # Using reparameterization trick to sample from a gaussian
        eps = Variable(torch.randn(batch_size, state_size)).to(device)
        return mu + torch.exp(log_var / 2) * eps

    def forward (self, x):
        # Encode
        conved = self.encoder(x)
        conved = conved.reshape(100, 100 * 2 * 2)
        mu = self.linear_encoder_mu(conved)
        sigma = self.linear_encoder_sigma(conved)

        state = self.linear_encoder(conved)
        # state = self.sample_z(mu, sigma)

        return state, mu, sigma

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.linear_decoder = nn.Sequential(
            nn.Linear(state_size + rp_size, 100 * 2 * 2), # b, 100, 2, 2
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(100, 200, 3, stride=2), # b, 200, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(200, 100, 5, stride=(4, 3), padding=1, output_padding=(1, 0)), # b, 100, 20, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(100, 3, (4, 3), stride=(4, 3), padding=1, output_padding=2), # b, 3, 80, 45
            nn.Sigmoid()
        )

    def forward (self, x):
        # Decode
        recon = self.linear_decoder(x)
        recon = recon.reshape(100, 100, 2, 2)
        recon = self.decoder(recon)
        return recon


def normalize_observation(observation):
    observation = observation.reshape(batch_size, img_size[0], img_size[1], 3)
    observation = np.transpose(observation, (0, 3, 2, 1))
    observation = observation.copy()
    observation = observation / 255.
    assert ((observation >= 0.).all() and (observation <= 1.).all())

    return observation

def write_to_tensorboard(writer, loss, recon_loss, kl_loss, current_batcher, step):
    if current_batcher < 10:
        writer.add_scalar("RP Reconstruction Loss", recon_loss, step)
        writer.add_scalar("RP Scaled Reconstruction Loss", (1. - beta) * recon_loss, step)
        writer.add_scalar("RP KL Loss", kl_loss, step)
        writer.add_scalar("RP Scaled KL Loss", beta * kl_loss, step)
        writer.add_scalar("RP Total Loss", loss, step)
    elif current_batcher == 10:
        writer.add_scalar("State Reconstruction Loss", recon_loss, step)
        writer.add_scalar("State Scaled Reconstruction Loss", (1. - beta) * recon_loss, step)
        writer.add_scalar("State KL Loss", kl_loss, step)
        writer.add_scalar("State Scaled KL Loss", beta * kl_loss, step)
        writer.add_scalar("State Total Loss", loss, step)

def save_weights(it, encoder, decoder, transition):
    if it % 10000 == 0:
        torch.save(encoder, "models/encoder_model_" + str(test_num) + "_" + str(it) + ".pt")
        torch.save(decoder, "models/decoder_model_" + str(test_num) + "_" + str(it) + ".pt")
        torch.save(transition, "models/transition_model_" + str(test_num) + "_" + str(it) + ".pt")


def pytorch_to_cv(img):
    input_numpy = img.detach().cpu().numpy()
    input_numpy = np.transpose(input_numpy, (2, 1, 0))
    input_numpy = np.round(input_numpy * 255.)
    input_numpy = input_numpy.astype(int)

    return input_numpy

def get_batch(starting_batch, ending_batch, batch_size):
    data_iter = starting_batch
    while data_iter <= ending_batch:
        data = np.load("training_data/training_data_" + str(data_iter) + ".npy")
        i = 0
        while i + batch_size < data.shape[0]:
            actions = data[i:i+batch_size, 0]
            observations = data[i:i+batch_size, 1:]
            yield observations, actions
            i += batch_size
        data_iter += 500
    yield None

def main():
    # Setup
    lr = 1e-2
    render_param_loss_term = 1e-3
    RPbatcher = get_batch(2050, 2100, batch_size)
    Sbatcher = get_batch(500, 1000, batch_size)
    current_batcher = 0

    # Make Networks objects
    RPen = RPencoder().to(device)
    Sen = Sencoder().to(device)
    decoder = Decoder().to(device)

    # Set solver
    RPparams = [x for x in RPen.parameters()]
    [RPparams.append(x) for x in decoder.parameters()]
    RPsolver = optim.Adam(RPparams, lr=lr)

    Sparams = [x for x in Sen.parameters()]
    [Sparams.append(x) for x in decoder.parameters()]
    Ssolver = optim.Adam(Sparams, lr=lr)

    # Main loop
    step = 0
    epoch = 0
    while True:
        # Solver setup
        RPsolver.zero_grad()
        Ssolver.zero_grad()

        if current_batcher < 10:
            batch = next(RPbatcher)
            if batch is None:
                epoch += 1
                RPbatcher = get_batch(2050, 2100, batch_size)
                batch = next(RPbatcher)
                # if epoch > 50:
                current_batcher += 1
                print(epoch, current_batcher)
        elif current_batcher == 10:
            batch = next(Sbatcher)
            if batch is None:
                epoch += 1
                Sbatcher = get_batch(500, 1000, batch_size)
                batch = next(Sbatcher)
                current_batcher = 0
                print(epoch, current_batcher)
        
        observations, actions = batch
        observations = normalize_observation(observations).astype(np.float32)
        observations = torch.from_numpy(observations).to(device)

        # Forward pass of the network
        render_params, rp_mu, rp_sigma = RPen(observations)
        state, s_mu, s_sigma = Sen(observations)
        encoded = torch.cat((render_params, state), 1)
        reconstructed_images = decoder(encoded)

        if step % 50 == 0:
            pics_dir = os.path.dirname("pics" + str(test_num) + "/")
            if not os.path.exists(pics_dir):
                os.makedirs(pics_dir)
            cv2.imwrite("pics" + str(test_num) + "/"+ str(step) + "original.jpg", pytorch_to_cv(observations[0]))
            cv2.imwrite("pics" + str(test_num) + "/" + str(step) + "reconstructed.jpg", pytorch_to_cv(reconstructed_images[0]))

        # Compute Loss
        assert ((reconstructed_images >= 0.).all() and (reconstructed_images <= 1.).all())

        recon_loss = F.binary_cross_entropy(reconstructed_images, observations)
        # kl_loss = 0.5 * torch.sum(torch.exp(rp_sigma) + rp_mu ** 2 - 1. - rp_sigma) if current_batcher == "renderparams" else 0.5 * torch.sum(torch.exp(s_sigma) + s_mu ** 2 - 1. - s_sigma)

        # loss = (1. - beta) * recon_loss + beta * kl_loss
        loss = recon_loss

        # Tensorboard
        write_to_tensorboard(writer, loss, recon_loss, 0., current_batcher, step)
        # Save weights
        # TODO: Save when we care about this

        # Backward pass
        loss.backward()

        # Update
        if current_batcher < 10:
            RPsolver.step()
        elif current_batcher == 10:
            Ssolver.step()
        step += 1
    

if __name__ == "__main__":
    main()