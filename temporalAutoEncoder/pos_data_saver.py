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
observations = []
states = []
actions = []
for i_episode in trange(0, 1001):
    # env.change_color()
    observation, state = env.reset()
    for t in range(1000):
        action = env.action_space.sample()
        observations.append(observation)
        states.append(state)
        actions.append(action)
        observation, state, reward, done, info = env.step(action)
        if done:
            break
observations = np.asarray(observations)
states = np.asarray(states)
actions = np.asarray(actions)
np.savez_compressed("pos_regressor_data", observations=observations, states=states, actions=actions)
env.close()
