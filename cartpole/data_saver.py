import gym
import gym_cartpole_visual
import numpy as np
from tqdm import trange
from skimage import io
import sys

from pyvirtualdisplay import Display

display = Display(visible=0, size=(100, 100))
display.start()

def main():
    env = gym.make("cartpole-visual-v1")
    imgs = []
    vels = []
    observation = env.reset()
    print(observation.shape)
    sys.exit()
    i = 0
    for _ in trange(100000):
        imgs.append(np.reshape(observation[:-2], (32, 32, 3)))
        vels.append(observation[-2:])
        action = np.random.randint(0, 2)
        observation, reward, done, info = env.step(action)

        if done:
            env.change_color(random=True)
            observation = env.reset()

        if len(imgs) > 10000:
            imgs = np.array(imgs)
            vels = np.array(vels)
            np.savez_compressed("training_data_small2/cartpole_vae_training_data" + str(i), images=imgs, vels=vels)

            i += 1
            imgs = []
            vels = []


    env.close()

if __name__ == "__main__":
    main()
