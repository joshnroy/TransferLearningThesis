"""A working example of deepmind_lab using python."""
from __future__ import absolute_import
# from __future__ import division
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
        print(action)
        action = self.convert_action(action)
        reward = self.env.step(action, num_steps=1)
        print(action)
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
        action_list = np.zeros(len(self.env.action_spec()), dtype=np.intc)
        if action_int / 1000000 == 2:
            action_list[0] = 10
        elif action_int / 1000000 == 1:
            action_list[0] = -10

        action_int %= 1000000

        if action_int / 100000 == 2:
            action_list[1] = 10
        elif action_int / 100000 == 1:
            action_list[1] = -10

        action_int %= 100000

        if action_int / 10000 == 2:
            action_list[2] = 1
        elif action_int / 10000 == 1:
            action_list[2] = -1

        action_int %= 10000

        if action_int / 1000 == 2:
            action_list[3] == 1
        elif action_int / 1000 == 1:
            action_list[3] = -1

        action_int %= 1000

        if action_int / 100 == 1:
            action_list[4] = 1

        action_int %= 100

        if action_int / 10 == 1:
            action_list[5] = 1

        if action_int == 1:
            action_list[6] = 1

        return action_list

if __name__ == '__main__':
    print("testing convert action function")
    env = SeekAvoidEnv()
    for i in range(2222111):
        print(i, env.convert_action(i))
