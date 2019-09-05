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
from tqdm import trange

from seekavoid_gymlike_wrapper import SeekAvoidEnv

import deepmind_lab

NUM_DATA_SAMPLES = 10000

def run():
    """Construct and start the environment."""

    env = SeekAvoidEnv()
    nb_actions = env.action_space.size

    FOLDER_NAME = "vae_training_data/"


    for i in trange(NUM_DATA_SAMPLES):
        action_list_episode = []
        reward_list_episode = []
        observations = env.reset()
        observations_list_episode = []
        done = False
        while not done:
            observations_list_episode.append(observations)
            action = np.random.randint(0, high=nb_actions)
            action_list_episode.append(action)
            observations, reward, done, _ = env.step(action)
            reward_list_episode.append(reward)
        observations_list_episode = np.array(observations_list_episode)
        action_list_episode = np.array(action_list_episode)
        reward_list_episode = np.array(reward_list_episode)
        np.savez_compressed(FOLDER_NAME + "episode_data" + str(i), actions=action_list_episode, rewards=reward_list_episode, observations=observations_list_episode)

if __name__ == '__main__':
    run()
