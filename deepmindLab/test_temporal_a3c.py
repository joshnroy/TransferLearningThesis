import numpy as np
import tensorflow as tf

import gym, time, random, threading, sys
from tqdm import trange
import csv
import matplotlib.pyplot as plt
import cv2

from skimage import io


from keras.models import *
from keras.layers import *
from keras import backend as K

from seekavoid_gymlike_wrapper import SeekAvoidEnv

import deepmind_lab

from temporal_a3c import Brain

def main():
    b = Brain(test=True)
    env = SeekAvoidEnv(test=True)

    e_rewards = []
    episodes = 300

    save_folder = "episode_images/"

    for i in trange(episodes):
        j = 0
        done = False
        obs = env.reset()
        obs = np.expand_dims(obs, axis=0)
        e_reward = 0.
        while not done:
            # io.imsave(save_folder + "img" + str(i) + "_" + str(j) + ".png", obs[0, :, :, :])
            prediction = b.predict_p(obs)[0]
            action = np.argmax(prediction)
            obs, reward, done, _ = env.step(action)
            obs = np.expand_dims(obs, axis=0)
            e_reward += reward
            j += 1
        e_rewards.append(e_reward)
    print(np.mean(e_rewards), np.std(e_rewards), np.max(e_rewards), np.min(e_rewards))
    # plt.hist(e_rewards, int(np.round(np.max(e_rewards))))
    # plt.show()

if __name__ == "__main__":
    main()
