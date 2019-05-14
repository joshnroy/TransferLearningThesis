"""A working example of deepmind_lab using python."""
from __future__ import absolute_import
# from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import six
import cv2
import huffman
from copy import deepcopy
from tqdm import trange, tqdm

import deepmind_lab

class SeekAvoidEnv():
    def __init__(self):
        config = {"width": "84", "height": "84", "fps": "60"}
        self.look_degree_step = 100
        self.observation_space = np.zeros((int(config['width']), int(config['height']), 3))

        self.i = 0

        level_script = "contributed/dmlab30/rooms_collect_good_objects_train"
        self.env = deepmind_lab.Lab(level_script, ['RGB_INTERLEAVED'], config, renderer='hardware')
        # self.window = cv2.namedWindow("Game Window", cv2.WINDOW_NORMAL)

        self.action_space = np.zeros(3)


    def seed(self, seed=None):
        return [seed]

    def step(self, action):
        # action = action.astype(np.intc)
        action = self.convert_action(action)
        reward = self.env.step(action, num_steps=10)
        done = not self.env.is_running()
        self.i += 1
        if not done:
            observations = np.asarray(self.env.observations()['RGB_INTERLEAVED']) / 255.
        else:
            observations = self.observation_space
        return observations, reward, done, {}

    def reset(self):
        self.env.reset()
        observations = np.asarray(self.env.observations()['RGB_INTERLEAVED']) / 255.
        return observations

    def render(self, mode):
        if self.env.is_running():
            image = self.env.observations()['RGB_INTERLEAVED']
            if (self.i > 500000):
                cv2.imwrite("images/img" + str(self.i) + ".png", image)
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
            action_list[0] = 25
        elif action_int == 1:
            action_list[0] = -25
        elif action_int == 2:
            action_list[3] = 1

        return action_list

    def save_img(self, name):
        img = self.render(None)
        cv2.imwrite("original_observations/" + name + ".png", img)

if __name__ == '__main__':
    env = SeekAvoidEnv()
    env.reset()
    env.save_img("test_img")
