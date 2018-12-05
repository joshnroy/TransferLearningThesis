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

display = Display(visible=0, size=(100, 100))
display.start()

writer = SummaryWriter("runs/initial")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

first_img = get_first_img()

img_size = (50, 75)
batch_size = 100
state_size = 1000
rp_size = 500
action_size = 1
image_dimension = img_size[0] * img_size[1] * 3
action_dimension = 2
hidden_dimension = 6 * 74 * 49
# c = 0
lr = 1e-6
beta = 1e-6
prediction_loss_term = 0.
loss_multiplier = 1.
render_param_loss_term = 0.001

# Encoder Network
class EncoderNet(torch.nn.Module):
    def __init__(self):
        super(EncoderNet, self).__init__()
        self.input_to_hidden = torch.nn.Conv2d(3, 6, 2)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.hidden_to_mu = torch.nn.Linear(hidden_dimension, state_size)
        self.hidden_to_var = torch.nn.Linear(hidden_dimension, state_size)

    def forward(self, x):
        h = self.input_to_hidden(x)
        h_flattened = torch.reshape(h, (batch_size, hidden_dimension))
        mu = self.hidden_to_mu(h_flattened)
        var = self.hidden_to_var(h_flattened)
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
        self.unpool = torch.nn.MaxUnpool2d(2, 2)
        self.hidden_to_reconstructed = torch.nn.ConvTranspose2d(6, 3, 2)
    
    def forward(self, z):
        h = nn.relu(self.state_to_hidden(z))
        h_unflattened = torch.reshape(h, (batch_size, 6, 74, 49))
        # h_unpooled = self.unpool(h_unflattened)
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
    if it % 1000 == 0:
        torch.save(encoder, "models/encoder_model_" + str(it) + ".pt")
        torch.save(decoder, "models/decoder_model_" + str(it) + ".pt")
        torch.save(transition, "models/transition_model_" + str(it) + ".pt")


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
    render_param_loss_f = torch.nn.MSELoss()
    # predicted_state = None

    # Main loop
    env = gym.make("cartpole-visual-v1")
    step = 0
    for i_episode in range(5000):
        observation = env.reset()
        for t in range(100):

            input_batch, actions = get_batch_and_actions(env, batch_size)

            # Forward pass of the network
            extracted_state, next_state, reconstructed_image, z_mu, z_var = forward_pass(T, input_batch, actions, encoder, decoder)
            extracted_state_with_action = torch.cat((extracted_state, actions), 1)

            if t % 50 == 0:
                cv2.imwrite("pics/"+ str(i_episode) + "_" + str(t) + "original.jpg", pytorch_to_cv(input_batch[0]))
                cv2.imwrite("pics/" + str(i_episode) + "_" + str(t) + "reconstructed.jpg", pytorch_to_cv(reconstructed_image[0]))

            # Compute Loss
            assert ((reconstructed_image >= 0.).all() and (reconstructed_image <= 1.).all())
            # whitevalmask = torch.ceil(np.ones(input_batch.shape, dtype=np.float32) - input_batch).to(device)

            recon_loss = nn.binary_cross_entropy(reconstructed_image, input_batch)
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var)

            render_param_loss = render_param_loss_f(extracted_state[0:batch_size - 1, 0:rp_size], extracted_state[1:batch_size, 0:rp_size])


            predicted_state = next_state[0:(batch_size - 1)]
            extracted_state_with_action = extracted_state_with_action[1:batch_size]
            prediction_loss = predicted_state_loss_f(predicted_state, extracted_state_with_action) / batch_size
            loss = ((1. - prediction_loss_term) * 
                        ((1. - render_param_loss_term) * ((1. - beta) * recon_loss + 
                            beta * kl_loss) + 
                            render_param_loss_term * render_param_loss) + 
                    prediction_loss_term * prediction_loss) * loss_multiplier
            write_to_tensorboard(writer, step, recon_loss, kl_loss, prediction_loss, render_param_loss, loss)


            # if predicted_state is not None:
            #     predicted_state = next_state[0:49]
            #     extracted_state_with_action = extracted_state_with_action[1:50]
            #     prediction_loss = predicted_state_loss_f(predicted_state, extracted_state_with_action)
            #     loss = ((1. - prediction_loss_term) * ((1. - beta) * recon_loss + beta * kl_loss) + prediction_loss_term * prediction_loss) * loss_multiplier
            #     write_to_tensorboard(writer, step, recon_loss, kl_loss, prediction_loss, loss)
            # else:
            #     loss = ((1. - beta) * recon_loss + beta * kl_loss) * loss_multiplier
            #     write_to_tensorboard(writer, step, recon_loss, kl_loss, None, loss)

            # Save weights
            save_weights(t, encoder, decoder, T)
            
            # Backward pass
            loss.backward(retain_graph=True)

            # Update
            # adaptive_lr = min((1/(10**(3/13000)))**step if step != 0 else 1., 1e-6)
            # for g in solver.param_groups:
            #     g['lr'] = adaptive_lr
            #     writer.add_scalar("Learning Rate", adaptive_lr, step)
            solver.step()
            step += 1

            print(i_episode, t)

            # predicted_state = next_state
    

if __name__ == "__main__":
    main()
