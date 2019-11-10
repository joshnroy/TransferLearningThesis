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
    def __init__(self, env, dqn_net, action_size, state_shape, learning_rate=1e-3, eps_start=0.9, eps_end=0.05, eps_decay=200, num_episodes=100, num_test_episodes=10, batch_size=128, update_interval=10, preprocessing_network=None):
        # Pytorch specific
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Environment related vars
        self.env = env
        self.actions = [torch.cuda.FloatTensor([i]) for i in range(action_size)]
        self.state_shape = state_shape
        self.gamma = 0.999

        self.preprocessing_network = preprocessing_network

        # Optimization parameters
        self.learning_rate = learning_rate
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.num_test_episodes = num_test_episodes

        # Networks
        self.dqn_net = dqn_net
        self.dqn_net = self.dqn_net.to(self.device)

        self.target_net = deepcopy(self.dqn_net)
        # self.target_net.load_state_dict(self.dqn_net.state_dict())
        self.target_net.eval()
        self.target_net = self.target_net.to(self.device)

        self.optimizer = optim.RMSprop(self.dqn_net.parameters(), lr=self.learning_rate)

        # Experience buffer
        self.memory = ReplayMemory(capacity=int(10000))


    def get_action(self, state, eps):
        if np.random.random() < eps:
            action = [np.random.randint(len(self.actions))]
            action = torch.cuda.FloatTensor(action)
        else:
            action = torch.cuda.FloatTensor([self.best_action(state)])

        return action

    def q_values(self, state, target=False):
        if not target:
            with torch.no_grad():
                return torch.stack([self.dqn_net(torch.cat([state, a]).to(self.device)) for a in self.actions])
        else:
            with torch.no_grad():
                return torch.stack([self.target_net(torch.cat([state, a]).to(self.device)) for a in self.actions])

    def best_action(self, state):
        action = torch.argmax(self.q_values(state))
        return action

    def train(self):
        self.dqn_net.train()
        self.target_net.eval()

        global_step = 0
        episode_rewards = []
        losses = []
        for i_e in trange(self.num_episodes):

            episode_reward = 0.
            state = self.env.reset()
            state = torch.cuda.FloatTensor(state)

            if self.preprocessing_network is not None:
                state = self.preprocessing_network(state)

            counter = 0
            while True:
                # Get and take action
                eps = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1. * global_step / self.eps_decay)
                action = self.get_action(state, eps)

                next_state, reward, done, _ = self.env.step(int(action.cpu().detach().item()))

                if self.preprocessing_network is not None:
                    next_state = self.preprocessing_network(next_state)

                next_state = torch.cuda.FloatTensor(next_state)
                reward = torch.cuda.FloatTensor([reward])
                done = torch.cuda.FloatTensor([done])

                # Bookkeeping
                global_step += 1
                counter += 1
                episode_reward += reward

                # Add to memory buffer
                self.memory.push([state, action, reward, next_state, done])
                state = next_state

                # Update target network
                if global_step % self.update_interval == 0:
                    # print("UPDATING")
                    self.target_net = deepcopy(self.dqn_net)
                    self.target_net.eval()

                # Train DQN
                # Q(s, a) = r + gamma * max_a' Q(s', a')
                if len(self.memory) >= self.batch_size:
                    sample = self.memory.sample(self.batch_size)

                    inputs = []
                    labels = []
                    for s, a, r, s_n, d in sample:
                        inputs.append(torch.cat([s, a]))
                        label = r
                        if d == 0.:
                            label += self.gamma * torch.max(self.q_values(s_n, target=True))
                        labels.append(label)

                    inputs = torch.stack(inputs)
                    labels = torch.stack(labels)
                    labels = labels.to(self.device)
                    # print(labels)

                    # inputs = torch.stack([torch.cat(x[:2]) for x in sample]) #s, a
                    # inputs = inputs.to(self.device)

                    # # Single DQN
                    # labels = torch.stack([r if d == 1. else r + self.gamma * torch.max(self.q_values(s_n, target=False)) for x in sample])
                    # # Double DQN
                    # # selected_state_action = torch.cat([x[3], torch.cuda.FloatTensor([self.best_action(x[3])])])
                    # # labels = torch.stack([x[2] if x[4] == 1. else x[2] + self.gamma * self.target_net(selected_state_action) for x in sample])

                    predictions = self.dqn_net(inputs).to(self.device)
                    # loss = F.mse_loss(predictions, labels)
                    loss = F.smooth_l1_loss(predictions, labels)

                    losses.append(loss)
                    # if i_e % 10 == 0:
                    #     print(loss)

                    self.optimizer.zero_grad()
                    loss.backward()
                    for param in self.dqn_net.parameters():
                        param.grad.data.clamp_(-1, 1)
                    self.optimizer.step()

                if done:
                    episode_rewards.append(episode_reward)

                    # if i_e % 10 == 0:
                    #     print("reward", episode_reward)

                    break


        return episode_rewards, losses

    def test(self):
        self.dqn_net.eval()

        global_step = 0
        episode_rewards = 0.
        losses = []
        for _ in trange(self.num_test_episodes):

            episode_reward = 0.
            state = self.env.reset()
            state = torch.cuda.FloatTensor(state)
            while True:
                # Get and take action
                print(self.q_values(state))
                action = self.get_action(state, 0.)

                self.env.render()

                next_state, reward, done, _ = self.env.step(int(action.cpu().detach().item()))

                next_state = torch.cuda.FloatTensor(next_state)
                reward = torch.cuda.FloatTensor([reward])
                done = torch.cuda.FloatTensor([done])

                # Bookkeeping
                global_step += 1
                episode_reward += reward
                if done:
                    episode_rewards += episode_reward
                    break

        return episode_rewards / self.num_test_episodes
