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

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPooling2D, Reshape, Permute, Activation, Conv3D, Lambda, Input
from keras.optimizers import Adam, RMSprop
import keras.backend as K

from keras.utils import multi_gpu_model

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

from factored_variational_autoencoder_deconv import vae

WEIGHTS_FILE = "factored_vae_deepmind_model7_factored.h5"

NUM_CONV_LAYERS = 3
NUM_FILTERS = 6
HIDDEN_SIZE = 48
NUM_HIDDEN_LAYERS = 5
WINDOW_LENGTH = 1

MULTI_GPU = False

def run():
    """Construct and start the environment."""

    env = SeekAvoidEnv()
    nb_actions = env.action_space.size # All possible action, where each action is a unit in this vector

    global vae
    print("#########################")
    input_shape = (WINDOW_LENGTH,) + env.observation_space.shape
    print(input_shape)
    in_layer = Input(shape=input_shape)
    reshaped = Reshape(target_shape=env.observation_space.shape)(in_layer)
    s_output_layer = Lambda(lambda x: x)(vae.layers[-2].outputs[2])
    # s_output_layer = Lambda(lambda x: x[:, 256:])(vae.layers[-2].outputs[2])
    vae = Model(vae.inputs, [s_output_layer])
    for layer in vae.layers:
        layer.trainable = False
    print(vae.summary())
    vae_output = vae(reshaped)

    x = Dense(256, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros')(vae_output)
    x = Dense(256, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros')(x)
    x = Dense(256, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros')(x)
    x = Dense(256, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros')(x)
    x = Dense(256, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros')(x)
    x = Dense(256, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros')(x)
    x = Dense(nb_actions, activation='linear')(x)
    model = Model(in_layer, [x])
    print(model.summary())
    if MULTI_GPU:
        model = multi_gpu_model(model, gpus=2)
        print(model.summary())

    num_warmup = 50000
    num_simulated_annealing = 300000 + num_warmup

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=num_simulated_annealing)

    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory, nb_steps_warmup=num_warmup, gamma=.99, target_model_update=10000, train_interval=4, delta_clip=1.)
    dqn.compile(Adam(lr=.25 * 1e-3), metrics=['mae'])

    history = dqn.fit(env, nb_steps=num_simulated_annealing + 150000, visualize=False, verbose=1)
    np.savez_compressed("vae_dqn_history4", episode_reward=np.asarray(history.history['episode_reward']))

    dqn.test(env, nb_episodes=10, visualize=True)

if __name__ == '__main__':
    run()
