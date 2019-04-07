import sys
import gym
import gym_cartpole_visual
import csv
import numpy as np
from tqdm import tqdm, trange

from keras.layers import Dense, Conv2D, BatchNormalization, MaxPool2D, Flatten, Activation, concatenate, Input, Lambda
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import keras

import pickle
import scipy

from copy import deepcopy

from pyvirtualdisplay import Display

display = Display(visible=0, size=(100, 100))
display.start()

EPISODES = 100000
RECORD_IMAGES = True

HISTORY_LEN = 5

NUM_FILTERS = 3
NUM_CONV_LAYERS = 3


# A2C(Advantage Actor-Critic) agent for the Cartpole
class A2CAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.actor_lr = 1. * 1e-3
        self.critic_lr = 5. * 1e-3

        # create model for policy network
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        if self.load_model:
            self.actor.load_weights("./save_model/cartpole_actor.h5")
            self.critic.load_weights("./save_model/cartpole_critic.h5")

    # approximate policy and value using Neural Network
    def build_actor(self):
        # Inputs
        inputs = [Input(shape=(64, 64, 3), name="input" + str(i) + "_actor") for i in range(HISTORY_LEN)]

        # Convolutional Layers
        conv = inputs
        numFilters = NUM_FILTERS
        for i in range(NUM_CONV_LAYERS):
            conv_layer = Conv2D(numFilters, 3)
            batchnorm_layer = BatchNormalization()
            maxpool_layer = MaxPool2D()
            conv = [maxpool_layer(batchnorm_layer(conv_layer(x))) for j, x in enumerate(conv)]
            numFilters *= 2

        # Flatten
        flattened = [Flatten()(x) for x in conv]

        # Concatenate
        output = concatenate(flattened)

        # output = Dense(160, activation='relu', kernel_initializer='he_uniform')(output)
        # output = Dense(96, activation='relu', kernel_initializer='he_uniform')(output)
        # output = Dense(48, activation='relu', kernel_initializer='he_uniform')(output)
        output = Dense(48, activation='relu')(output)
        output = Dense(48, activation='relu')(output)
        output = Dense(48, activation='relu')(output)
        output = Dense(self.action_size, activation='softmax')(output)
        # output = Lambda(lambda x: x * 500)(output)

        actor = Model(inputs = inputs, outputs=[output])

        actor.summary()

        # See note regarding crossentropy in cartpole_reinforce.py
        actor.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.actor_lr))
        return actor

    # critic: state is input and value of state is output of model
    def build_critic(self):
        # Inputs
        # inputs = Input(shape=(64, 64, 3), name="input_critic")

        # # Convolutional Layers
        # numFilters = NUM_FILTERS
        # conv = inputs
        # for i in range(NUM_CONV_LAYERS):
        #     conv_layer = Conv2D(numFilters, 3)
        #     batchnorm_layer = BatchNormalization()
        #     maxpool_layer = MaxPool2D()
        #     conv = maxpool_layer(batchnorm_layer(conv_layer(conv)))
        #     numFilters *= 2

        # Inputs
        inputs = [Input(shape=(64, 64, 3), name="input" + str(i) + "_actor") for i in range(HISTORY_LEN)]

        # Convolutional Layers
        conv = inputs
        numFilters = NUM_FILTERS
        for i in range(NUM_CONV_LAYERS):
            conv_layer = Conv2D(numFilters, 3)
            batchnorm_layer = BatchNormalization()
            maxpool_layer = MaxPool2D()
            conv = [maxpool_layer(batchnorm_layer(conv_layer(x))) for j, x in enumerate(conv)]
            numFilters *= 2

        # Flatten
        flattened = [Flatten()(x) for x in conv]

        # Concatenate
        output = concatenate(flattened)

        # output = Dense(160, activation='relu', kernel_initializer='he_uniform')(output)
        # output = Dense(96, activation='relu', kernel_initializer='he_uniform')(output)
        # output = Dense(48, activation='relu', kernel_initializer='he_uniform')(output)
        # output = Dense(48, activation='relu', kernel_initializer='he_uniform')(output)
        output = Dense(48, activation='relu')(output)
        output = Dense(48, activation='relu')(output)
        output = Dense(48, activation='relu')(output)
        output = Dense(self.value_size, activation='linear')(output)

        # output = Lambda(lambda x: x * 300)(output)

        critic = Model(inputs = inputs, outputs=[output])

        critic.summary()

        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
        return critic

    # using the output of policy network, pick action stochastically
    def get_action(self, states_history):
        if np.isnan(states_history).any():
            print("States history has a nan")
            sys.exit()
        policy = self.actor.predict(states_history)[0]
        # if np.isnan(policy).any():
        #     print(policy)
        #     print("Policy has a nan")
        # #     sys.exit()
        # if (np.sum(policy) <= 0 or np.isnan(policy).any()):
        #     normalized_policy = np.asarray([0.5, 0.5])
        # else:
        #     normalized_policy = policy / np.sum(policy)
        if np.isnan(policy).any():
            print("Concern")
            policy = [0.5, 0.5]
        softmax_policy = scipy.special.expit(policy) / np.sum(scipy.special.expit(policy))
        # if np.isnan(softmax_policy).any():
        #     print("policy", normalized_policy)
        #     print("softmax_policy", softmax_policy)
        #     print("exp(policy)", np.exp(normalized_policy))
        #     print("sum(exp(policy))", np.sum(np.exp(normalized_policy)))
        #     print("softmax_policy has a nan")
        #     softmax_policy = np.asarray([0.5, 0.5])
        #     # softmax_policy[(np.isfinite(softmax_policy))] = 0
        #     # softmax_policy[np.isnan(softmax_policy)] = 1
        #     # print("fixed softmax_policy", softmax_policy)
        #     # sys.exit()
        choice = np.random.choice(self.action_size, 1, p=softmax_policy)[0]
        # choice = np.argmax(policy)
        return choice

    # update policy network every episode
    def train_model(self, states_history, action, reward, next_state, done):
        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))

        value = self.critic.predict(states_history)[0]
        value = np.clip(value, -100, 500)
        assert(not np.isnan(value).any())

        next_states_history = states_history[1:]
        next_states_history.append(next_state)
        next_value = self.critic.predict(next_states_history)[0]
        next_value = np.clip(next_value, -100, 500)
        assert(not np.isnan(next_value).any())

        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action] = reward + self.discount_factor * (next_value) - value
            target[0][0] = reward + self.discount_factor * next_value
        softmax_advantages = advantages / np.sum(advantages)
        print(advantages, reward, self.discount_factor * (next_value), value)
        assert(np.sum(softmax_advantages) > 0.)

        assert(not np.isnan(states_history).any())
        # print(type(states_history), len(states_history), states_history[0].shape)
        actor_history = self.actor.fit(states_history, advantages, epochs=1, verbose=0)
        critic_history = self.critic.fit(states_history, target, epochs=1, verbose=0)

        print("critic", value, target, critic_history.history)
        actor_output = self.actor.predict(states_history)
        print("actor", actor_output, advantages, actor_history.history['loss'][0])
        assert(not np.isnan(actor_output).any())

        return actor_history.history['loss'][0], critic_history.history['loss'][0]


if __name__ == "__main__":

    # In case of CartPole-v1, maximum length of episode is 500
    env = gym.make('cartpole-visual-v1')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # make A2C agent
    agent = A2CAgent(state_size, action_size)

    scores, episodes = [], []

    with open('save_graph/visual_baseline5.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for e in trange(EPISODES):
            states_history = []

            done = False
            score = 0
            state = env.reset()
            state = state / 255.
            state = np.reshape(state, (1,) + state.shape)

            i = 0
            actor_losses = []
            critic_losses = []
            while not done:
                if agent.render:
                    env.render()

                if len(states_history) >= HISTORY_LEN:
                    states_history.pop(0)
                    states_history.append(state)
                    action = agent.get_action(states_history)
                    next_state, reward, done, info = env.step(action)
                    next_state = next_state / 255.


                    next_state = np.reshape(next_state, (1,) + next_state.shape)

                    # if an action make the episode end, then gives penalty of -100
                    reward = reward if not done or score == 499 else -100

                    actor_loss, critic_loss = agent.train_model(states_history, action, reward, next_state, done)
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)

                    score += reward
                    state = next_state
                else:
                    states_history.append(state)
                    action = np.random.choice(agent.action_size, 1)[0]
                    next_state, reward, done, info = env.step(action)
                    next_state = next_state / 255.

                    next_state = np.reshape(next_state, (1,) + next_state.shape)

                    # if an action make the episode end, then gives penalty of -100
                    reward = reward if not done or score == 499 else -100

                    score += reward
                    state = next_state

                if done:
                    # every episode, plot the play time
                    score = score if score == 500.0 else score + 100
                    scores.append(score)
                    episodes.append(e)
                    writer.writerow([str(e), str(score), str(np.mean(actor_losses)), str(np.mean(critic_losses))])
                    csvfile.flush()

                    actor_losses = []
                    critic_losses = []

                    # if the mean of scores of last 10 episode is bigger than 490
                    # stop training
                    if np.mean(scores[-min(10, len(scores)):]) > 490:
                        csvfile.close()
                        sys.exit()
                i += 1

            # save the model
            # if e % 50 == 0:
            #     agent.actor.save_weights("./save_model/cartpole_actor.h5")
            #     agent.critic.save_weights("./save_model/cartpole_critic.h5"
