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
from random import randint
import math

import gym
import gym_cartpole_visual

prev_runs = os.listdir("runs/")

trial_num = int(max(prev_runs)[5]) + 1

writer = SummaryWriter("runs/trial" + str(trial_num))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_size = (45, 80)
batch_size = 1
mixed_batch_size = 50
state_size = 4
rp_size = 50
action_size = 1
image_dimension = img_size[0] * img_size[1] * 3
action_dimension = 2

alpha = 0.5
beta = 1e-3
prediction_loss_term = 0.
reconstruction_weight_term = 0.5
encoding_regularization_term = 1e-2

# Network Hyperparameters
image_size = 64
conv_dim = 64
code_dim = 16
k_dim = 256
z_dim = 100

# Network
class rp_encoder(nn.Module):
    def __init__(self):
        super(rp_encoder, self).__init__()
        self.rp_encoder = nn.Sequential(
            nn.Conv2d(3, 20, (4, 3), stride=(4, 3)), # b, 200, 20, 15
            nn.ReLU(True),
            nn.MaxPool2d((4, 3), stride=(4, 3)), # b, 200, 5, 5
        )

        self.rp_linear_encoder = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(20 * 5 * 5, 10 * 5 * 5), # b, rp_size
            nn.ReLU(True),
            nn.Linear(10 * 5 * 5, rp_size)
        )

    def forward (self, x):
        # Encode
        conved = self.rp_encoder(x)
        conved = conved.reshape(mixed_batch_size, 20 * 5 * 5)
        render_params = self.rp_linear_encoder(conved)

        return render_params

class s_encoder(nn.Module):
    def __init__(self):
        super(s_encoder, self).__init__()

        # Compute Encoder layers
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(conv_dim))
        layers.append(nn.ReLU(True))
        
        # repeat_num = int(math.log2(image_size / code_dim))
        repeat_num = 3
        curr_dim = conv_dim
        for i in range(repeat_num):
            layers.append(nn.Conv2d(curr_dim, conv_dim * (i+2), kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(conv_dim * (i+2)))
            layers.append(nn.ReLU(True))
            curr_dim = conv_dim * (i+2)

        # Now we have (code_dim,code_dim,curr_dim)
        layers.append(nn.Conv2d(curr_dim, z_dim, kernel_size=1))

        # (code_dim,code_dim,z_dim)
        self.s_encoder = nn.Sequential(*layers)
        print(self.s_encoder)

        old_linear_size = z_dim * 10 * 5
        new_linear_size = int(np.floor(old_linear_size / 4))
        linear_layers = []
        linear_repeat_num = 5

        for _ in range(linear_repeat_num):
            linear_layers.append(nn.Linear(old_linear_size, new_linear_size))
            linear_layers.append(nn.ReLU(True))
            old_linear_size = new_linear_size
            new_linear_size = int(np.floor(old_linear_size / 4))

        linear_layers.append(nn.Linear(old_linear_size, state_size))

        self.s_linear_encoder = nn.Sequential(*linear_layers)
        print(self.s_linear_encoder)

    def forward (self, x):
        # Encode
        conved = self.s_encoder(x)
        conved = conved.reshape(mixed_batch_size, z_dim * 10 * 5)
        state = self.s_linear_encoder(conved)

        return state

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.linear_decoder = nn.Sequential(
            nn.Linear(state_size + rp_size, 12 * 5 * 5),
            nn.ReLU(True),
            nn.Linear(12 * 5 * 5, 23 * 5 * 5),
            nn.ReLU(True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(23, 23, 5, stride=(4, 3), padding=1, output_padding=(1, 0)), # b, 23, 23, 15
            nn.ReLU(True),
            nn.Dropout2d(p=0.2),
            nn.ConvTranspose2d(23, 3, (4, 3), stride=(4, 3), padding=1, output_padding=2), # b, 3, 80, 45
            nn.Sigmoid()
        )

    def forward (self, x):
        # Decode
        recon = self.linear_decoder(x)
        recon = recon.reshape(mixed_batch_size, 23, 5, 5)
        recon = self.decoder(recon)
        return recon


def normalize_observation(observation):
    observation = observation.reshape(mixed_batch_size, img_size[0], img_size[1], 3)
    observation = np.transpose(observation, (0, 3, 2, 1))
    observation = observation.copy()
    observation = observation / 255.
    assert ((observation >= 0.).all() and (observation <= 1.).all())

    return observation

def write_to_tensorboard(writer, loss, step):
    writer.add_scalar("Train Loss", loss, step)

def save_weights(it, encoder, decoder, transition):
    if it % 10000 == 0:
        torch.save(encoder, "models/encoder_model_" + str(trial_num) + "_" + str(it) + ".pt")
        torch.save(decoder, "models/decoder_model_" + str(trial_num) + "_" + str(it) + ".pt")
        torch.save(transition, "models/transition_model_" + str(trial_num) + "_" + str(it) + ".pt")


def pytorch_to_cv(img):
    img_channels = img

    return img_channels

def get_batch(starting_batch, ending_batch, batch_size, train):
    data_iter = starting_batch
    while data_iter <= ending_batch:
        data = None
        if train:
            data = np.load("training_data_with_state/training_data_with_state_" + str(data_iter) + ".npy")
        else:
            data = np.load("test_data/test_data_" + str(data_iter) + ".npy")
        i = 0
        while i + batch_size < data.shape[0]:
            observations = data[i:i+batch_size, :-4]
            states = data[i:i+batch_size, -4:][0]
            yield observations, states
            i += batch_size
        data_iter += 500
    yield None

def main():
    # Setup
    render_param_loss_term = 1e-3
    RPbatcher = get_batch(2050, 2100, batch_size, True)
    Sbatcher = get_batch(500, 1000, batch_size, True)
    current_batcher = 1
    # test_batcher = get_batch(500, 1000, mixed_batch_size, False)
    # test_observations, test_actions = next(test_batcher)
    # test_observations = normalize_observation(test_observations).astype(np.float32)
    # test_observations = torch.from_numpy(test_observations).to(device)

    # Make Networks objects
    rp_en = rp_encoder().to(device)
    s_en = s_encoder().to(device)
    decoder = Decoder().to(device)

    rp_en.train()
    s_en.train()
    decoder.train()

    lr = 1e-2
    # lr = 1e-8

    # Set solver
    rp_params = [x for x in rp_en.parameters()]
    rp_solver = optim.Adam(rp_params, lr=lr, weight_decay=0.)

    s_solver = optim.Adam(s_en.parameters(), lr=lr, weight_decay=0.)

    d_params = [x for x in decoder.parameters()]
    d_solver = optim.Adam(d_params, lr=lr, weight_decay=0.)


    # Main loop
    step = 0
    epoch = 0
    for _ in range(20000):
        print(step)
        # Solver setup
        rp_solver.zero_grad()
        s_solver.zero_grad()
        d_solver.zero_grad()

        mixed_batch = ([], [])
        for _ in range(mixed_batch_size):
            if current_batcher == 0:
                batch = next(RPbatcher)
                if batch is None:
                    epoch += 1
                    RPbatcher = get_batch(2050, 2100, batch_size, True)
                    batch = next(RPbatcher)
                current_batcher = 1
            elif current_batcher == 1:
                batch = next(Sbatcher)
                if batch is None:
                    epoch += 1
                    Sbatcher = get_batch(500, 1000, batch_size, True)
                    batch = next(Sbatcher)
                # current_batcher = 0
            mixed_batch[0].append(batch[0][0])
            mixed_batch[1].append(batch[1])
        mixed_batch = (np.array(mixed_batch[0]), np.array(mixed_batch[1]))
        # print("Got batch")

        # Shuffle batches
        # shuffle_rp = np.arange(int(mixed_batch_size / 2))
        # np.random.shuffle(shuffle_rp)
        # shuffle_s = np.arange(int(mixed_batch_size / 2))
        # np.random.shuffle(shuffle_s)
        
        observations, true_states = mixed_batch

        # observations[::2] = observations[::2][shuffle_rp]
        # true_states[::2] = true_states[::2][shuffle_rp]
        # observations[1::2] = observations[1::2][shuffle_s]
        # true_states[::2] = true_states[::2][shuffle_s]

        observations = normalize_observation(observations).astype(np.float32)
        observations = torch.from_numpy(observations).to(device)

        true_states = torch.from_numpy(true_states.astype(np.float32)).to(device)

        # Forward pass of the network
        # render_params = rp_en(observations)
        # print("Converted to CUDA")
        state = s_en(observations)
        # print("Forward Pass done")
        # encoded = torch.cat((render_params, state), 1)
        # reconstructed_images = decoder(encoded)

        # assert ((reconstructed_images >= 0.).all() and (reconstructed_images <= 1.).all())

        # if step % 1 == 0:
        #     writer.add_image("rp_original", pytorch_to_cv(observations[0]), step)
        #     writer.add_image("rp_reconstructed", pytorch_to_cv(reconstructed_images[0]), step)
            # writer.add_image("s_original", pytorch_to_cv(observations[1]), step)
        #     writer.add_image("s_reconstructed", pytorch_to_cv(reconstructed_images[1]), step)

        # if step % 100 == 0:
        #     for name, param in rp_en.named_parameters():
        #         writer.add_histogram(name, param.clone().cpu().data.numpy(), step)
        #     for name, param in s_en.named_parameters():
        #         writer.add_histogram(name, param.clone().cpu().data.numpy(), step)
        #     for name, param in decoder.named_parameters():
        #         writer.add_histogram(name, param.clone().cpu().data.numpy(), step)
        #     writer.add_histogram("S", state.clone().cpu().data.numpy(), step)

        # Compute Loss
        # rp_recon_loss = F.binary_cross_entropy(reconstructed_images[::2],
        #                                        observations[::2])
        # s_recon_loss = F.binary_cross_entropy(reconstructed_images[1::2],
        #                                       observations[1::2])

        # loss = alpha * rp_recon_loss + (1. - alpha) * s_recon_loss

        loss = F.mse_loss(state, true_states)
        # print("calculated loss")

        # Backward pass and Update
        loss.backward()
        # print("Did backward")
        # rp_solver.step()

        s_solver.step()
        # print("stepped")

        # d_solver.step()

        # Test the model
        # rp_en.eval()
        # s_en.eval()
        # decoder.eval()

        # # Forward pass of the network
        # render_params = rp_en(test_observations)
        # state = s_en(test_observations)
        # encoded = torch.cat((render_params, state), 1)
        # reconstructed_images = decoder(encoded)

        # # Compute Loss
        # assert ((reconstructed_images >= 0.).all() and (reconstructed_images <= 1.).all())

        # test_loss = F.binary_cross_entropy(reconstructed_images, test_observations)

        # if step % 100 == 0:
        #     writer.add_image("test_original", pytorch_to_cv(test_observations[randint(0,
        #                                                                               mixed_batch_size
        #                                                                               - 1)]), step)
        #     writer.add_image("test_reconstructed", pytorch_to_cv(reconstructed_images[randint(0,
        #                                                                                       mixed_batch_size
        #                                                                                       - 1)]), step)

        # rp_solver.zero_grad()
        # s_solver.zero_grad()
        # d_solver.zero_grad()

        # rp_en.train()
        # s_en.train()
        # decoder.train()

        # Tensorboard
        # print("writing to tensorboard")
        write_to_tensorboard(writer, loss, step)
        # print("wrote to tensorboard")
        # Save weights
        # TODO: Save when we care about this

        step += 1    

if __name__ == "__main__":
    main()