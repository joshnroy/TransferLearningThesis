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


NUM_DATA_SAMPLES = 100

def run():
    """Construct and start the environment."""

    env = SeekAvoidEnv()
    nb_actions = env.action_space.size

    for i in trange(NUM_DATA_SAMPLES):
        observations = env.reset()
        done = False
        j = 0
        while not done:
            cv2.imwrite("training_observations/obs_" + str(i) + "_" + str(j) + ".png", observations)
            # print(observations.shape)
            observations, reward, done, _ = env.step(np.random.randint(0, high=nb_actions))
            j += 1

if __name__ == '__main__':
    run()
