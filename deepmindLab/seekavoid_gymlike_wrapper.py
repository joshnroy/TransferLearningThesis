"""A working example of deepmind_lab using python."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import six
import cv2

import deepmind_lab

class SeekAvoidEnv():
    def __init__(self):
        config = {"width": "100", "height": "100"}
        self.observation_space = np.zeros((int(config['width']), int(config['height']), 3))

        level_script = "seekavoid_arena_01"
        self.env = deepmind_lab.Lab(level_script, ['RGB_INTERLEAVED'], config, renderer='hardware')

    def seed(self, seed=None):
        return [seed]

    def step(self, action):
        action = action.astype(np.intc)
        reward = self.env.step(action, num_steps=1)
        done = not self.env.is_running()
        if not done:
            observations = np.asarray(self.env.observations()['RGB_INTERLEAVED'])
        else:
            observations = self.observation_space
        return observations, reward, done, {}

    def reset(self):
        self.env.reset()
        observations = np.asarray(self.env.observations()['RGB_INTERLEAVED'])
        return observations

    def render(self):
        image = self.env.observations()['RGB_INTERLEAVED']
        return image

    def close(self):
        self.env.close()

    def convert_action(self, action_int):
        action_list = np.zeros(len(self.env.action_spec()))
        return action_list
