import gym
import gym_cartpole_visual

import numpy as np
import sys

from pyvirtualdisplay import Display

# from memory_profiler import profile

# fp = open('memory_profiler_without_close.log', 'w+')
# @profile(precision=10, stream=fp)
def main():
    display = Display(visible=0, size=(100, 100))
    display.start()
    array = []
    env = gym.make("cartpole-visual-v1")
    env.render()
    for i_episode in range(10001, 20001):
        for t in range(100):
            env.change_color()
            observation, _ = env.reset()

            action = env.action_space.sample()
            observation_flat = observation.flatten()
            array.append(np.insert(observation_flat, 0, action))
            del observation
            del observation_flat
        if i_episode > 0 and i_episode % 500 == 0:
            array = np.array(array)
            name = "rp_training_data/rp_training_data_" + str(i_episode) + ".npy"
            np.save(name, array)
            array = []
            print("Saved", name)
        print("Episode {} finished".format(i_episode))
    env.close()

if __name__ == "__main__":
    main()