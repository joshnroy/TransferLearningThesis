import gym
import numpy as np
from tqdm import trange

# from pyvirtualdisplay import Display

# display = Display(visible=0, size=(100, 100))
# display.start()

env = gym.make("CarRacing-v0")
observations = []
observation = env.reset()
for _ in trange(10000):
    observations.append(observation)
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)

    if done:
        observation = env.reset()

observations = np.array(observations)
np.savez_compressed('carracing_observations', observations=observations)
env.close()
