import sys
import gym
import gym_cartpole_visual
import csv
import numpy as np
from tqdm import tqdm, trange

from keras.layers import Dense, Conv2D, BatchNormalization, MaxPool2D, Flatten, Activation, concatenate, Input
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import keras

import pickle
import scipy

from pyvirtualdisplay import Display

from variational_autoencoder_deconv import vae

from copy import deepcopy

display = Display(visible=0, size=(100, 100))
display.start()

WEIGHTS_FILE = "vae_cnn_cartpole.h5"

EPISODES = 60000
RECORD_IMAGES = True

HISTORY_LEN = 10


# A2C(Advantage Actor-Critic) agent for the Cartpole
class A2CAgent:
    def __init__(self, state_size, action_size, vae):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 1.
        self.actor_lr = 1e-4 * 1.
        self.critic_lr = 1e-4 * 5.
        self.vae = vae

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

        # Pre-Trained Encoder
        vae = deepcopy(self.vae)
        encoded = [vae(inputs[i]) for i in range(HISTORY_LEN)]

        # Flatten
        # flattened = [Flatten()(encoded[i]) for i in range(HISTORY_LEN)]

        # Concatenate
        output = concatenate(encoded)

        output = Dense(160, activation='relu', kernel_initializer='he_uniform')(output)
        output = Dense(96, activation='relu', kernel_initializer='he_uniform')(output)
        output = Dense(96, activation='relu', kernel_initializer='he_uniform')(output)
        output = Dense(48, activation='relu', kernel_initializer='he_uniform')(output)
        output = Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform')(output)

        actor = Model(inputs = inputs, outputs=[output])

        actor.summary()

        # See note regarding crossentropy in cartpole_reinforce.py
        actor.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.actor_lr))
        return actor

    # critic: state is input and value of state is output of model
    def build_critic(self):
        # Inputs
        inputs = [Input(shape=(64, 64, 3), name="input" + str(i) + "_actor") for i in range(HISTORY_LEN)]

        # Pre-Trained Encoder
        vae = deepcopy(self.vae)
        encoded = [vae(inputs[i]) for i in range(HISTORY_LEN)]

        # Flatten
        # flattened = [Flatten()(encoded[i]) for i in range(HISTORY_LEN)]

        # Concatenate
        output = concatenate(encoded)

        output = Dense(160, activation='relu', kernel_initializer='he_uniform')(output)
        output = Dense(96, activation='relu', kernel_initializer='he_uniform')(output)
        output = Dense(96, activation='relu', kernel_initializer='he_uniform')(output)
        output = Dense(48, activation='relu', kernel_initializer='he_uniform')(output)
        output = Dense(self.value_size, activation='softmax', kernel_initializer='he_uniform')(output)

        critic = Model(inputs = inputs, outputs=[output])

        critic.summary()

        # See note regarding crossentropy in cartpole_reinforce.py
        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
        return critic

    # using the output of policy network, pick action stochastically
    def get_action(self, states_history):
        if np.isnan(states_history).any():
            print("States history has a nan")
            sys.exit()
        policy = self.actor.predict(states_history, batch_size=1).flatten()
        if np.isnan(policy).any():
            print("Policy has a nan")
            sys.exit()
        # print(policy)
        choice = np.random.choice(self.action_size, 1, p=policy)[0]
        # print(choice, policy)
        return choice

    # update policy network every episode
    def train_model(self, states_history, action, reward, next_state, done):
        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))

        value = self.critic.predict(states_history)[0]

        next_states_history = states_history[1:]
        next_states_history.append(next_state)
        next_value = self.critic.predict(next_states_history)[0]

        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action] = reward + self.discount_factor * (next_value) - value
            target[0][0] = reward + self.discount_factor * next_value

        self.actor.fit(states_history, advantages, epochs=1, verbose=0)
        self.critic.fit(states_history, target, epochs=1, verbose=0)


if __name__ == "__main__":

    vae.load_weights(WEIGHTS_FILE)

    model = Model(vae.inputs, [vae.layers[-2].outputs[2]])
    for layer in model.layers:
        layer.trainable = False

    # In case of CartPole-v1, maximum length of episode is 500
    env = gym.make('cartpole-visual-v1')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # make A2C agent
    agent = A2CAgent(state_size, action_size, model)

    scores, episodes = [], []

    with open('save_graph/visual_darla4_history10.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for e in trange(EPISODES):
            states_history = []

            done = False
            score = 0
            state = env.reset()
            state = state / 255.
            state = np.reshape(state, (1,) + state.shape)


            i = 0
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

                    agent.train_model(states_history, action, reward, next_state, done)

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
                    writer.writerow([str(e), str(score)])
                    csvfile.flush()

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
