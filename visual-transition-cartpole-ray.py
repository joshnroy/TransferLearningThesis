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

import ray
import ray.tune as tune

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
action_size = 1
image_dimension = img_size[0] * img_size[1] * 3
action_dimension = 2
hidden_dimension = 6 * 74 * 49

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
    # observation = cv2.resize(observation, (img_size[1], img_size[0]))
    observation = np.transpose(observation, (2, 1, 0))
    # observation = observation.reshape(image_dimension)
    observation = observation.copy()
    observation = observation / 255.
    assert ((observation >= 0.).all() and (observation <= 1.).all())

    return observation

def forward_pass(T, image, action, encoder, decoder):
    # Extract state, renderparams
    z_mu, z_var = encoder(image)
    state = sample_z(z_mu, z_var)

    # Decode image from state
    reconstructed_image = decoder(state)

    # Predict next_state
    next_state = T(torch.cat((state, action), 1))

    return state, next_state, reconstructed_image, z_mu, z_var

def pytorch_to_cv(img):
    input_numpy = img.detach().cpu().numpy()
    # input_reshaped = input_numpy.reshape(img_size[0], img_size[1], 3)
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

def train(config, reporter):
    # Setup
    import gym_cartpole_visual


    # Set Hyperparams
    rp_size = 250
    lr = config["lr"] #1e-5

    # Loss Hyperparams
    beta = config["beta"] ##0.8
    prediction_loss_term = 0.
    render_param_loss_term = config["render_param_loss_term"] #1e-2


    # Make transition Network
    T = TransitionNet().to(device)
    T.train()

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
    predicted_state_loss_f = torch.nn.MSELoss()
    render_param_loss_f = torch.nn.MSELoss()

    # Main loop
    env = gym.make("cartpole-visual-v1")
    step = 0
    while True:
        observation = env.reset()
        for _ in range(100):

            input_batch, actions = get_batch_and_actions(env, batch_size)

            # Forward pass of the network
            extracted_state, next_state, reconstructed_image, z_mu, z_var = forward_pass(T, input_batch, actions, encoder, decoder)
            extracted_state_with_action = torch.cat((extracted_state, actions), 1)

            # Compute Loss
            assert ((reconstructed_image >= 0.).all() and (reconstructed_image <= 1.).all())

            recon_loss = nn.binary_cross_entropy(reconstructed_image, input_batch)
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var)

            render_param_loss = render_param_loss_f(extracted_state[0:batch_size - 1, 0:rp_size], extracted_state[1:batch_size, 0:rp_size])
            render_param_loss = np.sum([render_param_loss_f(x, extracted_state[:, 0:rp_size]) for x in extracted_state[:, 0:rp_size]])

            predicted_state = next_state[0:(batch_size - 1)]
            extracted_state_with_action = extracted_state_with_action[1:batch_size]
            prediction_loss = predicted_state_loss_f(predicted_state, extracted_state_with_action) / batch_size

            loss = ((1. - prediction_loss_term) * 
                        ((1. - render_param_loss_term) * ((1. - beta) * recon_loss + 
                            beta * kl_loss) + 
                            render_param_loss_term * render_param_loss) + 
                    prediction_loss_term * prediction_loss)

            # Save weights
            reporter(step=step, recon_loss=recon_loss.cpu()[0], kl_loss=kl_loss.cpu()[0], prediction_loss=prediction_loss.cpu()[0], render_param_loss=render_param_loss.cpu(), mean_loss=loss.cpu())
            
            # Backward pass
            loss.backward(retain_graph=True)

            # Update
            solver.step()
            step += 1
    
def main():
    ray.init()

    all_trials = tune.run_experiments({
        "search_test": {
            "run": train,
            "stop": {"step": 100000},
            "config": {
                "lr": tune.grid_search([x * 1e-7 for x in range(1, 11)]),
                "beta": tune.grid_search([x * 1e-1 for x in range(1, 10)]),
                "render_param_loss_term": tune.grid_search([1. * 10 ** x for x in range(-5, 0)])
            },
            "trial_resources": {"cpu": 1, "gpu": 0.2},
            "local_dir":"~/Documents/Thesis/TransferLearningThesis/RayResults"
        }
        })

if __name__ == "__main__":
    main()
