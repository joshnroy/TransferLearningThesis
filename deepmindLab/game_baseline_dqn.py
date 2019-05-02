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

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPooling2D, Reshape, Permute, Activation, Conv3D
from keras.optimizers import Adam, RMSprop
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

NUM_CONV_LAYERS = 3
NUM_FILTERS = 6
HIDDEN_SIZE = 48
NUM_HIDDEN_LAYERS = 5
WINDOW_LENGTH = 1

def run():
    """Construct and start the environment."""

    env = SeekAvoidEnv()
    nb_actions = env.action_space.size # All possible action, where each action is a unit in this vector

    input_shape = (WINDOW_LENGTH,) + env.observation_space.shape
    model = Sequential()
    if WINDOW_LENGTH == 1:
        model.add(Reshape(target_shape=env.observation_space.shape, input_shape = input_shape))
    # if K.image_dim_ordering() == 'tf':
    #     # (width, height, channels)
    #     model.add(Permute((2, 3, 1), input_shape=input_shape))
    # elif K.image_dim_ordering() == 'th':
    #     # (channels, width, height)
    #     model.add(Permute((1, 2, 3), input_shape=input_shape))
    # else:
    #     raise RuntimeError('Unknown image_dim_ordering.')
    if WINDOW_LENGTH == 1:
        model.add(Conv2D(8, (3, 3), strides=(2, 2), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (3, 3), strides=(2, 2)))
        model.add(Activation('relu'))
    else:
        model.add(Conv3D(8, (3, 3, 1), strides=(2, 2, 1), data_format='channels_first', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv3D(16, (3, 3, 1), strides=(2, 2, 1), data_format='channels_first'))
        model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    # model = Sequential()
    # num_filters = NUM_FILTERS
    # model.add(Reshape(target_shape = env.observation_space.shape, input_shape=(1,) + env.observation_space.shape))

    # model.add(Conv2D(num_filters, 3))
    # model.add(MaxPooling2D())
    # model.add(BatchNormalization())
    # num_filters *= 2

    # for _ in range(NUM_CONV_LAYERS - 1):
    #     model.add(Conv2D(num_filters, 3))
    #     model.add(MaxPooling2D())
    #     model.add(BatchNormalization())
    #     num_filters *= 2

    # model.add(Flatten())
    # for _ in range(NUM_HIDDEN_LAYERS):
    #     model.add(Dense(HIDDEN_SIZE, activation='relu'))
    # model.add(Dense(nb_actions, activation='sigmoid'))
    # print(model.summary())

    num_warmup = 50000
    num_simulated_annealing = 300000 + num_warmup

    memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=num_simulated_annealing)

    dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory, nb_steps_warmup=num_warmup, gamma=.99, target_model_update=10000, train_interval=4, delta_clip=1.)
    dqn.compile(Adam(lr=.00025), metrics=['mae'])
    # dqn.compile(RMSprop(lr=.00025), metrics=['mae'])

    history = dqn.fit(env, nb_steps=num_simulated_annealing + 150000, visualize=False, verbose=1)
    np.savez_compressed("dqn_RMSprop_visual_history", episode_reward=np.asarray(history.history['episode_reward']))

    dqn.test(env, nb_episodes=10, visualize=True)

if __name__ == '__main__':
    run()
