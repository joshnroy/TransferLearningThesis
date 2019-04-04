import gym
import gym_cartpole_visual

import numpy as np
from tqdm import trange
import cv2

from pyvirtualdisplay import Display

display = Display(visible=0, size=(100, 100))
display.start()

array = []
env = gym.make("cartpole-visual-v1")
for i_episode in trange(0, 101):
    observation, state = env.reset()
    for t in range(1000):
        cv2.imwrite("sparse_training_data/sparse_training_data_" + str(i_episode) + "_" + str(t) + ".jpg", observation)
        action = env.action_space.sample()
        # observation_flat = observation.flatten()
        # array.append(np.insert(observation_flat, 0, action))
        observation, state, reward, done, info = env.step(action)
        if done:
            # if i_episode > 0 and i_episode % 500 == 0:
            #     # array = np.array(array)
            #     # name = "sparse_training_data/sparse_training_data_" + str(i_episode) + ".npy"
            #     # np.save(name, array)
            #     del array
            #     array = []
            #     print("Saved", name)
            # print("Episode {} finished".format(i_episode))
            break
env.close()
