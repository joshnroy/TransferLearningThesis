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
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1) # b, 8, 2, 2
        )

        self.decoder = nn.Sequential(
                        nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward (self, x):
        # Encode
        x = self.encoder(x)
        
        # Decode
        x = self.decoder(x)
        return x

def normalize_observation(observation):
    # observation = cv2.resize(observation, (img_size[1], img_size[0]))
    observation = np.transpose(observation, (2, 1, 0))
    # observation = observation.reshape(image_dimension)
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
    lr = 1e-3

    # Make transition Network

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
    env.reset()
    for i_episode in range(5000):
        print(str(step))
        for t in range(100):
            # Solver setup
            solver.zero_grad()

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

            recon_loss = nn.binary_cross_entropy(reconstructed_image, input_batch)
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var)

            render_param_loss = torch.sqrt(render_param_loss_f(extracted_state[0:batch_size - 1, 0:rp_size], extracted_state[1:batch_size, 0:rp_size]))

            predicted_state = next_state[0:(batch_size - 1)]
            extracted_state_with_action = extracted_state_with_action[1:batch_size]
            prediction_loss = predicted_state_loss_f(predicted_state, extracted_state_with_action) / batch_size
            loss = ((1. - prediction_loss_term) * 
                        ((1. - render_param_loss_term) * ((1. - beta) * recon_loss + 
                            beta * kl_loss) + 
                            render_param_loss_term * render_param_loss) + 
                    prediction_loss_term * prediction_loss)
            write_to_tensorboard(writer, step, recon_loss, kl_loss, prediction_loss, render_param_loss, loss)
            writer.add_scalar("renderparam sum", torch.sum(torch.abs(extracted_state[0, 0:rp_size])), step)

            # Save weights
            save_weights(t, encoder, decoder, T)
            
            # Backward pass
            loss.backward(retain_graph=True)

            # Adaptive LR
            lr = lr if recon_loss > 0.15 else 1e-4
            for g in solver.param_groups:
                g['lr'] = lr
                writer.add_scalar("Learning Rate", lr, step)

            # Update
            solver.step()
            step += 1
    

if __name__ == "__main__":
    main()
