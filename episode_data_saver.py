import gym
import gym_cartpole_visual

import numpy as np

from pyvirtualdisplay import Display

display = Display(visible=0, size=(100, 100))
display.start()

array = []
for i_episode in range(1001, 2001):
    env = gym.make("cartpole-visual-v1")
    observation = env.reset()
    for t in range(100):
        action = env.action_space.sample()
        observation_flat = observation.flatten()
        array.append(np.insert(observation_flat, 0, action))
        observation, reward, done, info = env.step(action)
        if done:
            if i_episode > 0 and i_episode % 500 == 0:
                array = np.array(array)
                name = "training_data/training_data_" + str(i_episode) + ".npy"
                np.save(name, array)
                array = []
                print("Saved", name)
            print("Episode {} finished".format(i_episode))
            break