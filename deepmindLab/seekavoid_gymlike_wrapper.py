"""A working example of deepmind_lab using python."""
from __future__ import absolute_import
# from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import six
import cv2
from copy import deepcopy
from tqdm import trange, tqdm

import deepmind_lab

class SeekAvoidEnv():
    def __init__(self, test=False):
        config = {"width": "84", "height": "84", "fps": "60", "allowHoldOutLevels": "true"}
        self.look_degree_step = 100
        self.observation_space = np.zeros((int(config['width']), int(config['height']), 3))

        self.i = 0
        self.step_increment = 10

        level_script = "contributed/dmlab30/rooms_collect_good_objects_train" if not test else "contributed/dmlab30/rooms_collect_good_objects_test"
        # level_script = "contributed/dmlab30/rooms_collect_good_objects_train" if not test else "contributed/dmlab30/rooms_collect_good_objects_train"
        self.env = deepmind_lab.Lab(level_script, ['RGB_INTERLEAVED'], config, renderer='hardware')

        self.action_space = np.zeros(3)


    def seed(self, seed=None):
        return [seed]

    def num_steps(self):
        return self.i * self.step_increment

    def step(self, action):
        action = self.convert_action(action)
        reward = self.env.step(action, num_steps=self.step_increment)
        env_done = not self.env.is_running()
        self.i += 1
        if not env_done:
            observations = np.asarray(self.env.observations()['RGB_INTERLEAVED']) / 255.
        else:
            self.env.reset()
            observations = np.asarray(self.env.observations()['RGB_INTERLEAVED']) / 255.

        done = self.num_steps() >= 3600
        return observations, reward, done, {}

    def reset(self):
        self.i = 0
        self.env.reset()
        observations = np.asarray(self.env.observations()['RGB_INTERLEAVED']) / 255.
        return observations

    def render(self, mode):
        if self.env.is_running():
            image = self.env.observations()['RGB_INTERLEAVED']
        else:
            image = self.observation_space
        # cv2.imshow("Game Window", image)
        # cv2.waitKey(1)
        return image

    def close(self):
        self.env.close()

    def convert_action(self, action_int):
        action_list = np.zeros(7, np.intc)

        if action_int == 0:
            action_list[0] = 50
        elif action_int == 1:
            action_list[0] = -50
        elif action_int == 2:
            action_list[0] = 50
            action_list[3] = 1
        elif action_int == 3:
            action_list[0] = -50
            action_list[3] = 1
        elif action_int == 4:
            action_list[0] = -1
        elif action_int == 5:
            action_list[0] = 1
        elif action_int == 6:
            action_list[1] = -1
        elif action_int == 7:
            action_list[1] = 1

        return action_list

    def save_img(self, name):
        img = self.render(None)
        cv2.imwrite("original_observations/" + name + ".png", img)

if __name__ == '__main__':
    env = SeekAvoidEnv()
    env.reset()
    env.save_img("test_img")
