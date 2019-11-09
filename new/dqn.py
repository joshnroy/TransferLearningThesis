import torch
import random
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from copy import deepcopy

from tqdm import trange

import matplotlib.pyplot as plt

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition # a transition should be (s, a, r, s', done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN():
    def __init__(self, env, dqn_net, action_size, state_shape, learning_rate=1e-3, eps=0.1, num_episodes=100, num_test_episodes=10, batch_size=200, update_interval=500):
        # Pytorch specific
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Environment related vars
        self.env = env
        self.actions = [torch.FloatTensor([i]) for i in range(action_size)]
        self.state_shape = state_shape
        self.gamma = 0.95

        # Optimization parameters
        self.learning_rate = learning_rate
        self.eps = eps
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.num_test_episodes = num_test_episodes

        # Networks
        self.dqn_net = dqn_net
        self.target_net = deepcopy(self.dqn_net)
        self.target_net.load_state_dict(self.dqn_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.dqn_net.parameters(), lr=self.learning_rate)

        # Experience buffer
        self.memory = ReplayMemory(capacity=10000)


    def get_action(self, state, eps):
        if np.random.random_sample((1,))[0] < eps:
            action = [np.random.randint(len(self.actions))]
            action = torch.FloatTensor(action)
        else:
            action = torch.FloatTensor([self.best_action(state)])

        return action

    def q_values(self, state):
        return torch.stack([self.dqn_net(torch.cat([state, a])) for a in self.actions])

    def best_action(self, state):
        action = torch.argmax(self.q_values(state))
        return action

    def train(self):
        self.dqn_net.train()
        self.target_net.eval()

        global_step = 0
        episode_rewards = []
        losses = []
        for _ in trange(self.num_episodes):

            episode_reward = 0.
            state = self.env.reset()
            state = torch.FloatTensor(state)
            while True:
                # Get and take action
                action = self.get_action(state, self.eps)

                next_state, reward, done, _ = self.env.step(int(action.cpu().detach().item()))

                next_state = torch.FloatTensor(next_state)
                reward = torch.FloatTensor([reward])
                done = torch.FloatTensor([done])

                # Bookkeeping
                global_step += 1
                episode_reward += reward
                if done:
                    episode_rewards.append(episode_reward)
                    break

                # Add to memory buffer
                self.memory.push([state, action, reward, next_state, done])
                state = next_state

                # Update target network
                if global_step % self.update_interval:
                    self.target_net.load_state_dict(self.dqn_net.state_dict())

                # Train DQN
                # Q(s, a) = r + gamma * max_a' Q(s', a')
                if len(self.memory) >= self.batch_size:
                    sample = self.memory.sample(self.batch_size)
                    inputs = torch.stack([torch.cat(x[:2]) for x in sample]) #s, a
                    labels = torch.stack([x[2] if x[4] == 1. else x[2] + self.gamma * torch.max(self.q_values(x[3])) for x in sample])

                    predictions = self.dqn_net(inputs)
                    loss = F.mse_loss(predictions, labels)

                    losses.append(loss)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

        return episode_rewards, losses

    def test(self):
        self.dqn_net.eval()

        global_step = 0
        episode_rewards = 0.
        losses = []
        for _ in trange(self.num_test_episodes):

            episode_reward = 0.
            state = self.env.reset()
            state = torch.FloatTensor(state)
            while True:
                # Get and take action
                action = self.get_action(state, 0.)

                next_state, reward, done, _ = self.env.step(int(action.cpu().detach().item()))

                next_state = torch.FloatTensor(next_state)
                reward = torch.FloatTensor([reward])
                done = torch.FloatTensor([done])

                # Bookkeeping
                global_step += 1
                episode_reward += reward
                if done:
                    episode_rewards += episode_reward
                    break

        return episode_rewards / self.num_test_episodes
