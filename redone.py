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
state_size = 1000
rp_size = 10
action_size = 1
image_dimension = img_size[0] * img_size[1] * 3
action_dimension = 2
hidden_dimension = 3 * 73 * 49

beta = 0.1
prediction_loss_term = 0.
render_param_loss_term = 0.

# Network
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, (4, 3), stride=(4, 3)), # b, 16, 20, 15
            nn.ReLU(True),
            nn.MaxPool2d((4, 3), stride=(4, 3)), # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1), # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1) # b, 8, 2, 2
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2), # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=(4, 3), padding=1, output_padding=(1, 0)), # b, 8, 20, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, (4, 3), stride=(4, 3), padding=1, output_padding=2), # b, 3, 80, 45
            nn.Sigmoid()
        )

    def forward (self, x):
        # Encode
        state = self.encoder(x)
        
        # Decode
        recon = self.decoder(state)
        return state, recon

def normalize_observation(observation):
    observation = observation.reshape(batch_size, img_size[0], img_size[1], 3)
    observation = np.transpose(observation, (0, 3, 2, 1))
    observation = observation.copy()
    observation = observation / 255.
    assert ((observation >= 0.).all() and (observation <= 1.).all())

    return observation

def write_to_tensorboard(writer, it, recon_loss, kl_loss, prediction_loss, render_param_loss, total_loss):
    writer.add_scalar("Reconstruction Loss", recon_loss, it)
    writer.add_scalar("Scaled Reconstruction Loss", (1. - prediction_loss_term) * (1. - beta) * recon_loss, it)
    writer.add_scalar("KL Loss", kl_loss, it)
    writer.add_scalar("Scaled KL Loss", (1. - prediction_loss_term) * (1. - render_param_loss_term) * beta * kl_loss, it)
    writer.add_scalar("RP Loss", render_param_loss, it)
    writer.add_scalar("Scaled RP Loss", (1. - prediction_loss_term) * render_param_loss_term * render_param_loss, it)
    if prediction_loss is not None:
        writer.add_scalar("Prediction Loss", prediction_loss, it)
        writer.add_scalar("Scaled Prediction Loss", prediction_loss_term * prediction_loss, it)
    writer.add_scalar("Total Loss", total_loss, it)

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

def get_batch(batch_size):
    data_iter = 500
    while data_iter <= 1000:
        data = np.load("training_data/training_data_" + str(data_iter) + ".npy")
        i = 0
        while i + batch_size < data.shape[0]:
            actions = data[i:i+batch_size, 0]
            observations = data[i:i+batch_size, 1:]
            yield observations, actions
            i += batch_size
        data_iter += 500

def main():
    # Setup
    lr = 1e-3
    batcher = get_batch(batch_size)

    # Make transition Network
    ae = autoencoder().to(device)

    # Set solver
    solver = optim.Adam(ae.parameters(), lr=lr)

    # Losses
    predicted_state_loss_f = torch.nn.MSELoss()
    render_param_loss_f = torch.nn.MSELoss()

    # Main loop
    step = 0
    for i_episode in range(5000):
        print(str(step))
        # Solver setup
        solver.zero_grad()

        observations, actions = next(batcher)
        observations = normalize_observation(observations).astype(np.float32)
        observations = torch.from_numpy(observations).to(device)

        # Forward pass of the network
        extracted_state, reconstructed_images = ae(observations)

        if i_episode % 50 == 0:
            pics_dir = os.path.dirname("pics" + str(test_num) + "/")
            if not os.path.exists(pics_dir):
                os.makedirs(pics_dir)
            cv2.imwrite("pics" + str(test_num) + "/"+ str(i_episode) + "original.jpg", pytorch_to_cv(observations[0]))
            cv2.imwrite("pics" + str(test_num) + "/" + str(i_episode) + "reconstructed.jpg", pytorch_to_cv(reconstructed_images[0]))

        # Compute Loss
        assert ((reconstructed_images >= 0.).all() and (reconstructed_images <= 1.).all())

        recon_loss = F.binary_cross_entropy(reconstructed_images, observations)
        # kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var) TODO: May need to make network variational later

        render_param_loss = torch.sqrt(render_param_loss_f(extracted_state[0:batch_size - 1, 0:rp_size], extracted_state[1:batch_size, 0:rp_size]))

        loss = (1. - render_param_loss_term) * recon_loss + render_param_loss_term * render_param_loss

        # Tensorboard
        write_to_tensorboard(writer, step, recon_loss, 0., 0., render_param_loss, loss)
        writer.add_scalar("renderparam sum", torch.sum(torch.abs(extracted_state[0, 0:rp_size])), step)

        # Save weights
        # TODO: Save when we care about this

        # Backward pass
        loss.backward()

        # Adaptive LR

        # Update
        solver.step()
        step += 1
    

if __name__ == "__main__":
    main()
