import gym
import gym_cartpole_visual
import cv2


for i_episode in range(200):
    env = gym.make("cartpole-visual-v1")
    observation = env.reset()
    for t in range(100):
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        cv2.imwrite("randpics/observation_" + str(i_episode) + "_" + str(t) + ".jpg", observation)
        if done:
            print("Episode {} finished".format(i_episode))
            break
