import gym
import numpy as np
from tqdm import trange
from skimage import io
import sys

from pyvirtualdisplay import Display

# display = Display(visible=0, size=(100, 100))
# display.start()

def main():
    env = gym.make("Pendulum-v0")
    imgs = []
    observation = env.reset()
    env.change_color()
    observation = env.reset()
    i = 0
    for _ in trange(100001):
        imgs.append(observation)
        # io.imshow(observation)
        # io.show()
        # env.render()
        action = [np.random.random() * 4. - 2.]
        observation, reward, done, info = env.step(action)

        if done:
            env.change_color()
            observation = env.reset()

        if len(imgs) > 10000:
            imgs = np.array(imgs)
            np.savez_compressed("vae_pendulum_data/arr" + str(i), images=imgs)

            i += 1
            imgs = []


    env.close()

if __name__ == "__main__":
    main()
