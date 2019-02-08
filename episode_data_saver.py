import gym
import gym_cartpole_visual

import numpy as np

from pyvirtualdisplay import Display

display = Display(visible=0, size=(100, 100))
display.start()

array = []
for i_episode in range(0, 1001):
    env = gym.make("cartpole-visual-v1")
    observation, state = env.reset()
    for t in range(100):
        action = env.action_space.sample()
        observation_flat = observation.flatten()
        array.append(np.concatenate([observation_flat, state]))
        observation, state, reward, done, info = env.step(action)
        if done:
            if i_episode > 0 and i_episode % 500 == 0:
                array = np.array(array)
                name = "training_data_with_state/training_data_with_state_" + str(i_episode) + ".npy"
                np.save(name, array)
                array = []
                print("Saved", name)
            print("Episode {} finished".format(i_episode))
            break
