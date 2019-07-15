import numpy as np
import tensorflow as tf

import gym, time, random, threading, sys
from skimage import io

from keras.models import *
from keras.layers import *
from keras import backend as K

from tqdm import trange
import csv

from seekavoid_gymlike_wrapper import SeekAvoidEnv

import deepmind_lab

from darla_a3c import Brain

def main():
    b = Brain(test=True)
    env = SeekAvoidEnv(test=True)

    e_rewards = []
    episodes = 300

    for _ in trange(episodes):
        done = False
        obs = env.reset()
        start_time = env.num_steps()
        obs = np.expand_dims(obs, axis=0)
        e_reward = 0.
        while not done:
            prediction = b.predict_p(obs)[0]
            action = np.argmax(prediction)
            obs, reward, done, _ = env.step(action)
            # print(obs.shape, obs.max(), obs.mean(), obs.min())
            # io.imshow(obs)
            # io.show()
            # sys.exit()
            obs = np.expand_dims(obs, axis=0)
            e_reward += reward
        end_time = env.num_steps()
        # print("Episode took", end_time - start_time, "with reward", e_reward)
        e_rewards.append(e_reward)
    print(np.mean(e_rewards), np.std(e_rewards), np.max(e_rewards), np.min(e_rewards))

if __name__ == "__main__":
    main()
