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
        config = {"width": "40", "height": "40"}
        self.look_degree_step = 100
        self.observation_space = np.zeros((int(config['width']), int(config['height']), 3))

        self.i = 0


        level_script = "seekavoid_arena_01"
        self.env = deepmind_lab.Lab(level_script, ['RGB_INTERLEAVED'], config, renderer='hardware')
        # self.window = cv2.namedWindow("Game Window", cv2.WINDOW_NORMAL)

# Create huffman encoding of actions
        # possible_actions = []
        # print("Adding ", self.env.action_spec()[0]['name'])

        # for i in range(self.env.action_spec()[0]['min'], self.env.action_spec()[0]['max'], self.look_degree_step):
        # for i in range(-200, 200, self.look_degree_step):
        #     possible_actions.append([i])
        # print(len(possible_actions))

        # x = self.env.action_spec()[1]
        # print("Adding ", x['name'], len(possible_actions))
        # possible_actions_copy = possible_actions
        # possible_actions = []
        # for short_action in possible_actions_copy:
        #     for i in range(-200, 200, self.look_degree_step):
        #         to_add = deepcopy(short_action)
        #         to_add.append(i)
        #         possible_actions.append(to_add)

        # for x in self.env.action_spec()[2:]:
        #     print(x['name'], len(possible_actions), x['min'], x['max'], range(x['min'], x['max']+1))
        #     possible_actions_copy = possible_actions
        #     possible_actions = []
        #     for short_action in possible_actions_copy:
        #         for i in range(x['min'], x['max'] + 1):
        #             to_add = deepcopy(short_action)
        #             to_add.append(i)
        #             possible_actions.append(to_add)
        # possible_actions_tuples = []
        # for x in possible_actions:
        #     assert(len(x) == 7)
        #     possible_actions_tuples.append(tuple(x))

        # self.codebook = {i: x for i, x, in enumerate(possible_actions_tuples)}
        # self.action_space = np.zeros(len(possible_actions_tuples), dtype=np.intc)
        # self.int_codebook = self.codebook

        self.action_space = np.zeros(10)


    def seed(self, seed=None):
        return [seed]

    def step(self, action):
        action = action.astype(np.intc)
        action = self.convert_action(action)
        reward = self.env.step(action, num_steps=1)
        done = not self.env.is_running()
        self.i += 1
        if not done:
            observations = np.asarray(self.env.observations()['RGB_INTERLEAVED'])
        else:
            observations = self.observation_space
        return observations, reward, done, {}

    def reset(self):
        self.env.reset()
        observations = np.asarray(self.env.observations()['RGB_INTERLEAVED'])
        return observations

    def render(self, mode):
        if self.env.is_running():
            image = self.env.observations()['RGB_INTERLEAVED']
            if (self.i > 1125000):
                cv2.imwrite("original_observations/img" + str(self.i) + ".png", image)
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
            action_list[1] = 25
        elif action_int == 3:
            action_list[1] = -25
        elif action_int == 4:
            action_list[2] = 1
        elif action_int == 5:
            action_list[2] = -1
        elif action_int == 6:
            action_list[3] = 1
        elif action_int == 7:
            action_list[3] = -1
        elif action_int == 8:
            action_list[4] = 1
        elif action_int == 9:
            action_list[5] = 1
        elif action_int == 10:
            action_list[6] = 1

        return action_list

if __name__ == '__main__':
    print("testing convert action function")
    env = SeekAvoidEnv()
