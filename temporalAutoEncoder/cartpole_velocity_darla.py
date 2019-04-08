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

from variational_autoencoder_deconv import vae

display = Display(visible=0, size=(100, 100))
display.start()

WEIGHTS_FILE = "vae_cnn_cartpole.h5"

EPISODES = 100000
RECORD_IMAGES = True

HISTORY_LEN = 5

NUM_FILTERS = 3
NUM_CONV_LAYERS = 3


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
        self.discount_factor = 0.99
        self.actor_lr = 1. * 1e-5
        self.critic_lr = 5. * 1e-5
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
        img_input = Input(shape=(64, 64, 3), name="input_img_actor")
        vel_input = Input(shape=(2,), name="input_vel_actor")
        inputs = [img_input, vel_input]

        # Pre-Trained encoder
        vae = deepcopy(self.vae)
        encoded = vae(img_input)

        # Concatenate
        output = concatenate([encoded, vel_input])

        output = Dense(24, activation='relu')(output)
        output = Dense(24, activation='relu')(output)
        output = Dense(24, activation='relu')(output)
        output = Dense(self.action_size, activation='softmax')(output)

        actor = Model(inputs = inputs, outputs=[output])

        actor.summary()

        # See note regarding crossentropy in cartpole_reinforce.py
        actor.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.actor_lr))
        return actor

    # critic: state is input and value of state is output of model
    def build_critic(self):
        # Inputs
        img_input = Input(shape=(64, 64, 3), name="input_img_actor")
        vel_input = Input(shape=(2,), name="input_vel_actor")
        inputs = [img_input, vel_input]

        # Pre-Trained encoder
        vae = deepcopy(self.vae)
        encoded = vae(img_input)

        # Concatenate
        output = concatenate([encoded, vel_input])

        output = Dense(24, activation='relu')(output)
        output = Dense(24, activation='relu')(output)
        output = Dense(24, activation='relu')(output)
        output = Dense(self.value_size, activation='linear')(output)

        critic = Model(inputs = inputs, outputs=[output])

        critic.summary()

        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
        return critic

    # using the output of policy network, pick action stochastically
    def get_action(self, combined_input):
        if np.isnan(combined_input[0]).any() or np.isnan(combined_input[1]).any():
            print("States history has a nan")
            sys.exit()
        policy = self.actor.predict(combined_input)[0]
        if np.isnan(policy).any():
            print("Concern")
            policy = [0.5, 0.5]
        softmax_policy = scipy.special.expit(policy) / np.sum(scipy.special.expit(policy))
        choice = np.random.choice(self.action_size, 1, p=softmax_policy)[0]
        return choice

    # update policy network every episode
    def train_model(self, combined_input, action, reward, next_combined_input, done):
        target = np.zeros((1, self.value_size))
        advantages = np.zeros((1, self.action_size))

        value = self.critic.predict(combined_input)[0]
        value = np.clip(value, -100, 500)
        assert(not np.isnan(value).any())

        next_value = self.critic.predict(next_combined_input)[0]
        next_value = np.clip(next_value, -100, 500)
        assert(not np.isnan(next_value).any())

        if done:
            advantages[0][action] = reward - value
            target[0][0] = reward
        else:
            advantages[0][action] = reward + self.discount_factor * (next_value) - value
            target[0][0] = reward + self.discount_factor * next_value
        softmax_advantages = advantages / np.sum(advantages)
        if np.sum(softmax_advantages) <= 0.:
            softmax_advantages = np.asarray([0.5, 0.5])

        assert(not np.isnan(combined_input[0]).any() and not np.isnan(combined_input[1]).any())
        actor_history = self.actor.fit(combined_input, advantages, epochs=1, verbose=0)
        critic_history = self.critic.fit(combined_input, target, epochs=1, verbose=0)

        # print("critic", value, target)
        actor_output = self.actor.predict(combined_input)
        # print("actor", actor_output, advantages)
        assert(not np.isnan(actor_output).any())

        return actor_history.history['loss'][0], critic_history.history['loss'][0]


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

    with open('save_graph/velocity_darla1.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for e in trange(EPISODES):
            states_history = []

            done = False
            score = 0
            state, true_state = env.reset()
            velocity = np.asarray([true_state[1], true_state[3]])
            state = state / 255.
            state = np.reshape(state, (1,) + state.shape)
            velocity = np.reshape(velocity, (1,) + velocity.shape)

            i = 0
            actor_losses = []
            critic_losses = []
            while not done:
                if agent.render:
                    env.render()

                combined_input = [state, velocity]

                action = agent.get_action(combined_input)
                next_state, next_true_state, reward, done, info = env.step(action)
                next_velocity = np.asarray([next_true_state[1], next_true_state[3]])
                next_state = next_state / 255.


                next_state = np.reshape(next_state, (1,) + next_state.shape)
                next_velocity = np.reshape(next_velocity, (1,) + next_velocity.shape)
                next_combined_input = [next_state, next_velocity]

                # if an action make the episode end, then gives penalty of -100
                reward = reward if not done or score == 499 else -100

                actor_loss, critic_loss = agent.train_model(combined_input, action, reward, next_combined_input, done)
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

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
