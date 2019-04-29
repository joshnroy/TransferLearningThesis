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
from keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPooling2D, Reshape
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

NUM_CONV_LAYERS = 3
NUM_FILTERS = 6
HIDDEN_SIZE = 24
NUM_HIDDEN_LAYERS = 3


def run():
    """Construct and start the environment."""

    env = SeekAvoidEnv()
    nb_actions = env.action_space.size # All possible action, where each action is a unit in this vector

    model = Sequential()
    num_filters = NUM_FILTERS
    model.add(Reshape(target_shape = env.observation_space.shape, input_shape=(1,) + env.observation_space.shape))
    model.add(Conv2D(num_filters, 3))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    num_filters *= 2
    for _ in range(NUM_CONV_LAYERS - 1):
        model.add(Conv2D(num_filters, 3))
        model.add(MaxPooling2D())
        model.add(BatchNormalization())
        num_filters *= 2
    model.add(Flatten())
    for _ in range(NUM_HIDDEN_LAYERS):
        model.add(Dense(HIDDEN_SIZE, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    print(model.summary())

    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()

    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100, enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

    dqn.test(env, nb_episodes=5, visualize=False)

if __name__ == '__main__':
    run()