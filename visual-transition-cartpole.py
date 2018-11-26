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

import gym
import gym_cartpole_visual

from pyvirtualdisplay import Display

display = Display(visible=0, size=(1920, 1080))
display.start()

writer = SummaryWriter("runs/initial")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

first_img = get_first_img()

img_size = (50, 75)
batch_size = 1
state_size = 10
rp_size = 5
action_size = 1
image_dimension = img_size[0] * img_size[1] * 3
action_dimension = 2
hidden_dimension = 1000
# c = 0
lr = 1e-6
beta = 0.2
prediction_loss_term = 0.1
loss_multiplier = 100.

# Encoder Network
class EncoderNet(torch.nn.Module):
    def __init__(self):
        super(EncoderNet, self).__init__()
        self.input_to_hidden = torch.nn.Linear(image_dimension, hidden_dimension)
        self.hidden_to_mu = torch.nn.Linear(hidden_dimension, state_size)
        self.hidden_to_var = torch.nn.Linear(hidden_dimension, state_size)

    def forward(self, x):
        h = nn.relu(self.input_to_hidden(x))
        mu = self.hidden_to_mu(h)
        var = self.hidden_to_var(h)
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
        self.input_to_hidden = torch.nn.Linear(state_size, hidden_dimension)
        self.hidden_to_reconstructed = torch.nn.Linear(hidden_dimension, image_dimension)
    
    def forward(self, z):
        h = nn.relu(self.input_to_hidden(z))
        X = nn.sigmoid(self.hidden_to_reconstructed(h))
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
    observation = cv2.resize(observation, (img_size[1], img_size[0]))
    observation = observation.reshape(image_dimension)
    observation = Variable(torch.from_numpy(observation.copy())).to(device)
    observation = observation.float() / 255.
    assert ((observation >= 0.).all() and (observation <= 1.).all())

    return observation

def forward_pass(T, image, action, encoder, decoder):
    # Extract state, renderparams
    z_mu, z_var = encoder(image)
    state = sample_z(z_mu, z_var)

    # Decode image from state
    reconstructed_image = decoder(state)

    # Predict next_state
    next_state = T(torch.cat((state, torch.tensor([[action]], dtype=torch.float).to(device)), 1))

    return state, next_state, reconstructed_image, z_mu, z_var

def write_to_tensorboard(writer, it, recon_loss, kl_loss, prediction_loss, total_loss):
    writer.add_scalar("Reconstruction Loss", recon_loss, it)
    writer.add_scalar("Scaled Reconstruction Loss", (1. - prediction_loss_term) * (1. - beta) * recon_loss, it)
    writer.add_scalar("KL Loss", kl_loss, it)
    writer.add_scalar("Scaled KL Loss", (1. - prediction_loss_term) * beta * kl_loss, it)
    if prediction_loss is not None:
        writer.add_scalar("Prediction Loss", prediction_loss, it)
        writer.add_scalar("Scaled Prediction Loss", prediction_loss_term * prediction_loss, it)
    writer.add_scalar("Total Loss", total_loss, it)

def save_weights(it, encoder, decoder, transition):
    if it % 100 == 0:
        torch.save(encoder, "models/encoder_model_" + str(it) + ".pt")
        torch.save(decoder, "models/decoder_model_" + str(it) + ".pt")
        torch.save(transition, "models/transition_model_" + str(it) + ".pt")


def pytorch_to_cv(img):
    input_numpy = img.detach().cpu().numpy()
    input_reshaped = input_numpy.reshape(img_size[0], img_size[1], 3)
    input_reshaped = input_reshaped[...,::-1]
    input_reshaped = np.round(input_reshaped * 255.)
    input_reshaped = input_reshaped.astype(int)

    return input_reshaped

def main():
    # Setup

    # Make transition Network
    T = TransitionNet().to(device)
    T.train()

    encoder = EncoderNet().to(device)
    encoder.train()
    decoder = DecoderNet().to(device)
    decoder.train()

    # Make Autoencoder Network
    # params = [Wxh, bxh, Whz_mu, bhz_mu, Whz_var, bhz_var, Wzh, bzh, Whx, bhx]

    # Set solver
    params = []
    params += [x for x in encoder.parameters()]
    params += [x for x in decoder.parameters()]
    params += [x for x in T.parameters()]
    solver = optim.Adam(params, lr=lr)

    # Losses
    predicted_state_loss_f = torch.nn.MSELoss()
    predicted_state = None

    # Main loop
    env = gym.make("cartpole-visual-v1")
    step = 0
    for i_episode in range(5000):
        observation = env.reset()
        for t in range(100):
            # Take a random action
            action = env.action_space.sample()

            observation, reward, done, info = env.step(action)

            # Run the simulation
            input_image = normalize_observation(observation)

            # Forward pass of the network
            extracted_state, next_state, reconstructed_image, z_mu, z_var = forward_pass(T, input_image, action, encoder, decoder)

            if i_episode % 100 == 0:
                cv2.imwrite("pics/"+ str(i_episode) + "_" + str(t) + "original.jpg", pytorch_to_cv(input_image))
                cv2.imwrite("pics/" + str(i_episode) + "_" + str(t) + "reconstructed.jpg", pytorch_to_cv(reconstructed_image))

            # Compute Loss
            assert ((reconstructed_image >= 0.).all() and (reconstructed_image <= 1.).all())
            recon_loss = nn.binary_cross_entropy(reconstructed_image, input_image)
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var)
            if predicted_state is not None:
                prediction_loss = predicted_state_loss_f(predicted_state, torch.cat((extracted_state, torch.tensor([[action]], dtype=torch.float).to(device)), 1))
                loss = ((1. - prediction_loss_term) * ((1. - beta) * recon_loss + beta * kl_loss) + prediction_loss_term * prediction_loss) * loss_multiplier
                write_to_tensorboard(writer, step, recon_loss, kl_loss, prediction_loss, loss)
            else:
                loss = ((1. - beta) * recon_loss + beta * kl_loss) * loss_multiplier
                write_to_tensorboard(writer, step, recon_loss, kl_loss, None, loss)

            # Save weights
            save_weights(t, encoder, decoder, T)
            
            # Backward pass
            loss.backward(retain_graph=True)

            # Update
            adaptive_lr = 1. / step if step != 0 else 1.
            for g in solver.param_groups:
                g['lr'] = adaptive_lr
                writer.add_scalar("Learning Rate", adaptive_lr, step)
            solver.step()
            step += 1

            predicted_state = next_state

            if done:
                print("Episode {} finished".format(i_episode))
                break
    

if __name__ == "__main__":
    main()
