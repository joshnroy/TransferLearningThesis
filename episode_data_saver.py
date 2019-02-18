import gym
import gym_cartpole_visual

import numpy as np
    
from pyvirtualdisplay import Display

display = Display(visible=0, size=(100, 100))
display.start()

array = []
env = gym.make("cartpole-visual-v1")
for i_episode in range(10001, 20001):
    observation, state = env.reset()
    for t in range(1000):
        action = env.action_space.sample()
        observation_flat = observation.flatten()
        array.append(np.insert(observation_flat, 0, action))
        del observation
        del observation_flat
        observation, state, reward, done, info = env.step(action)
        if done:
            if i_episode > 0 and i_episode % 500 == 0:
                array = np.array(array)
                name = "state_training_data/state_training_data_" + str(i_episode) + ".npy"
                np.save(name, array)
                del array
                array = []
                print("Saved", name)
            print("Episode {} finished".format(i_episode))
            break
env.close()