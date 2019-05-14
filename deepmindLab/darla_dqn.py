## Copyright (C) 2016-17 Google Inc.
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License along
## with this program; if not, write to the Free Software Foundation, Inc.,
## 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
################################################################################
"""A working example of deepmind_lab using python."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import six
import cv2

from seekavoid_gymlike_wrapper import SeekAvoidEnv

import deepmind_lab

from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPooling2D, Reshape, Permute, Activation, Conv3D, Lambda, Input, Concatenate
from keras.optimizers import Adam, RMSprop
from keras.initializers import RandomUniform
import keras.backend as K

from keras.utils import multi_gpu_model

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from variational_autoencoder_deconv import vae

WEIGHTS_FILE = "../temporalAutoEncoder/darla_beta_vae.h5"

HIDDEN_SIZE = 256
NUM_HIDDEN_LAYERS = 5
WINDOW_LENGTH = 4

MULTI_GPU = False

def run():
    """Construct and start the environment."""

    env = SeekAvoidEnv()
    nb_actions = env.action_space.size # All possible action, where each action is a unit in this vector

    global vae
    vae.load_weights(WEIGHTS_FILE)
    print("#########################")
    original_input = Input(shape=(WINDOW_LENGTH,) + env.observation_space.shape)
    in_layer = [Lambda(lambda x: x[:, i, :, :])(original_input) for i in range(WINDOW_LENGTH)]
    vae = Model(vae.inputs, [vae.layers[-2].outputs[2]])
    for layer in vae.layers:
        layer.trainable = False
    print(vae.summary())
    vae_output = [vae(x) for x in in_layer]

    x = Concatenate()(vae_output)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(nb_actions, activation='linear')(x)
    model = Model(original_input, [x])
    print(model.summary())
    if MULTI_GPU:
        model = multi_gpu_model(model, gpus=2)
        print(model.summary())

    num_warmup = 50000
    num_simulated_annealing = 300000 + num_warmup

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=num_simulated_annealing)

    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory, nb_steps_warmup=num_warmup, gamma=.99, target_model_update=10000, train_interval=4, delta_clip=1.)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])

    history = dqn.fit(env, nb_steps=num_simulated_annealing + 150000, visualize=False, verbose=1)
    dqn.save_weights("darla_dqn_weights")
    np.savez_compressed("darla_dqn_history", episode_reward=np.asarray(history.history['episode_reward']))

    dqn.test(env, nb_episodes=10, visualize=True)

if __name__ == '__main__':
    run()
