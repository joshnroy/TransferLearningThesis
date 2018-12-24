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
from cartpole_image_interface import read_next_batch, get_first_img
import sys

import gym
import gym_cartpole_visual

from pyvirtualdisplay import Display

display = Display(visible=0, size=(100, 100))
display.start()

prev_runs = os.listdir("runs/")

test_num = int(max(prev_runs)[4]) + 1

writer = SummaryWriter("runs/test" + str(test_num))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

first_img = get_first_img()

img_size = (50, 75)
batch_size = 100
state_size = 1000
rp_size = 250
action_size = 1
image_dimension = img_size[0] * img_size[1] * 3
action_dimension = 2
hidden_dimension = 6 * 74 * 49

beta = 0.1
render_param_loss_term = 1e-5

# Encoder Network
class EncoderNet(torch.nn.Module):
    def __init__(self):
        super(EncoderNet, self).__init__()
        self.input_to_hidden = torch.nn.Conv2d(3, 6, 2)
        self.hidden_to_mu = torch.nn.Linear(hidden_dimension, state_size)
        self.hidden_to_var = torch.nn.Linear(hidden_dimension, state_size)

    def forward(self, x):
        h = nn.relu(self.input_to_hidden(x))
        h_flattened = torch.reshape(h, (batch_size, hidden_dimension))
        mu = nn.relu(self.hidden_to_mu(h_flattened))
        var = nn.relu(self.hidden_to_var(h_flattened))
        return mu, var

# Sample from encoder network
def sample_z(mu, log_var):
    # Using reparameterization trick to sample from a gaussian
    eps = Variable(torch.randn(batch_size, state_size)).to(device)
    return mu + torch.exp(log_var / 2) * eps

# Decoder Network
class DecoderNet(torch.nn.Module):
    def __init__(self):
        super(DecoderNet, self).__init__()
        self.state_to_hidden = torch.nn.Linear(state_size, hidden_dimension)
        self.hidden_to_reconstructed = torch.nn.ConvTranspose2d(6, 3, 2)
    
    def forward(self, z):
        h = nn.relu(self.state_to_hidden(z))
        h_unflattened = torch.reshape(h, (batch_size, 6, 74, 49))
        X = torch.sigmoid(self.hidden_to_reconstructed(h_unflattened))
        return X

# Transition Network
class TransitionNet(torch.nn.Module):
    def __init__(self):
        super(TransitionNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_size + action_size, state_size + action_size)
        self.fc2 = torch.nn.Linear(state_size + action_size, 100)
        self.fc3 = torch.nn.Linear(100, state_size + action_size)

    def forward(self, s):
        s = nn.relu(self.fc1(s))
        s = nn.relu(self.fc2(s))
        s = self.fc3(s)
        return s

def normalize_observation(observation):
    observation = np.transpose(observation, (2, 1, 0))
    observation = observation.copy()
    observation = observation / 255.
    assert ((observation >= 0.).all() and (observation <= 1.).all())

    return observation

def forward_pass(T, image, action, encoder, decoder):
    # Extract state, renderparams
    z_mu, z_var = encoder(image)
    state = sample_z(z_mu, z_var)

    # Predict next_state
    next_state = T(torch.cat((state, action), 1))

    # Decode image from state
    reconstructed_next_image = decoder(next_state[0:batch_size, 0:state_size])

    return state, next_state, reconstructed_next_image, z_mu, z_var

def write_to_tensorboard(writer, it, recon_loss, kl_loss, render_param_loss, total_loss):
    writer.add_scalar("Reconstruction Loss", recon_loss, it)
    writer.add_scalar("Scaled Reconstruction Loss", (1. - beta) * recon_loss, it)
    writer.add_scalar("KL Loss", kl_loss, it)
    writer.add_scalar("Scaled KL Loss", (1. - render_param_loss_term) * beta * kl_loss, it)
    writer.add_scalar("RP Loss", render_param_loss, it)
    writer.add_scalar("Scaled RP Loss", render_param_loss_term * render_param_loss, it)
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

def get_batch_and_actions(env, batch_size):
    batch = []
    actions = []

    for _ in range(batch_size):
        # Take a random action
        action = env.action_space.sample()
        actions.append([action])

        # Run the simulation
        observation, reward, done, info = env.step(action)

        # Normalize Observation
        input_image = normalize_observation(observation)
        batch.append(input_image)

        if done:
            observation = env.reset()
    
    batch = torch.from_numpy(np.array(batch, dtype=np.float32)).to(device)
    actions = torch.from_numpy(np.array(actions, dtype=np.float32)).to(device)

    return batch, actions

def main():
    # Setup
    lr = 1e-4

    # Make transition Network
    T = TransitionNet().to(device)
    T.train()

    # Make Autoencoder Network
    encoder = EncoderNet().to(device)
    encoder.train()
    decoder = DecoderNet().to(device)
    decoder.train()

    # Set solver
    params = []
    params += [x for x in encoder.parameters()]
    params += [x for x in decoder.parameters()]
    params += [x for x in T.parameters()]
    solver = optim.Adam(params, lr=lr)

    # Losses
    render_param_loss_f = torch.nn.MSELoss()

    # Main loop
    env = gym.make("cartpole-visual-v1")
    step = 0
    for i_episode in range(5000):
        print(str(step))
        observation = env.reset()
        for t in range(100):

            input_batch, actions = get_batch_and_actions(env, batch_size)

            # Forward pass of the network
            extracted_state, next_state, reconstructed_image, z_mu, z_var = forward_pass(T, input_batch, actions, encoder, decoder)
            extracted_state_with_action = torch.cat((extracted_state, actions), 1)

            if t % 50 == 0:
                pics_dir = os.path.dirname("pics" + str(test_num) + "/")
                if not os.path.exists(pics_dir):
                    os.makedirs(pics_dir)
                cv2.imwrite("pics" + str(test_num) + "/"+ str(i_episode) + "_" + str(t) + "original.jpg", pytorch_to_cv(input_batch[0]))
                cv2.imwrite("pics" + str(test_num) + "/" + str(i_episode) + "_" + str(t) + "reconstructed.jpg", pytorch_to_cv(reconstructed_image[0]))

            # Compute Loss
            assert ((reconstructed_image >= 0.).all() and (reconstructed_image <= 1.).all())

            recon_loss = nn.binary_cross_entropy(reconstructed_image[0:(batch_size-1)], input_batch[1:batch_size])
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var)

            render_param_loss = render_param_loss_f(extracted_state[0:batch_size - 1, 0:rp_size], extracted_state[1:batch_size, 0:rp_size])

            loss = (1. - render_param_loss_term) * (beta * kl_loss + (1. - beta) * recon_loss) + render_param_loss_term * render_param_loss
            write_to_tensorboard(writer, step, recon_loss, kl_loss, render_param_loss, loss)
            writer.add_scalar("renderparam sum", torch.sum(torch.abs(extracted_state[0, 0:rp_size])), step)

            # Save weights
            save_weights(t, encoder, decoder, T)
            
            # Backward pass
            loss.backward(retain_graph=True)

            # Change Learning Rate
            lr = lr if recon_loss > 0.2 else 1e-6
            if step > 2000:
                lr = 1e-7
            for g in solver.param_groups:
                g['lr'] = lr
                writer.add_scalar("Learning Rate", lr, step)

            # Update
            solver.step()
            step += 1
    

if __name__ == "__main__":
    main()
