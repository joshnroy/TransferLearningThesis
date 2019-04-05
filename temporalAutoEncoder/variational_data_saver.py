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
observation = env.reset()
for i_episode in trange(0, 101):
    env.change_color()
    observation = env.reset()
    for t in range(1000):
        cv2.imwrite("training_data/training_data_" + str(i_episode) + "_" + str(t) + ".jpg", observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            break
env.close()
