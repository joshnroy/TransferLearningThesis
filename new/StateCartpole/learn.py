import torch
import random
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from copy import deepcopy
from tqdm import trange
import matplotlib.pyplot as plt
import gym
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from dqn import DQN, ReplayMemory

class DQN_network(nn.Module):
    def __init__(self):
        super(DQN_network, self).__init__()
        self.fc1 = nn.Linear(5, 48)
        self.fc2 = nn.Linear(48, 48)
        self.fc3 = nn.Linear(48, 1)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    env = gym.make('CartPole-v0')

    dqn_net = DQN_network()
    dqn_agent = DQN(env, dqn_net, 2, 4)

    training_rewards, training_losses = dqn_agent.train()

    plt.plot(training_rewards)
    plt.title("Training Rewards")
    plt.show()

    plt.plot(training_losses)
    plt.title("Training Losses")
    plt.show()

    test_reward = dqn_agent.test()
    print("Testing Reward", test_reward)

if __name__ == "__main__":
    main()
