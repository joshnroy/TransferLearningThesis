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

import dnnlib
import dnnlib.tflib as tflib
import config
import pickle
import scipy

from pyvirtualdisplay import Display

display = Display(visible=0, size=(100, 100))
display.start()
keras.losses.custom_loss = keras.losses.mean_squared_error

MODEL_NAME = "reinforcement/AgainHighLowIndependence"

EPISODES = 100000
RECORD_IMAGES = True


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
        self.discount_factor = 1.
        self.actor_lr = 1e-4 * 1.
        self.critic_lr = 1e-4 * 5.

        # create model for policy network
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        if self.load_model:
            self.actor.load_weights("./save_model/cartpole_actor.h5")
            self.critic.load_weights("./save_model/cartpole_critic.h5")

    # approximate policy and value using Neural Network
    # actor: state is input and probability of each action is output of model
    def build_actor_thesis(self):
        # Get the RP extractor pre-trained model
        encoder_model = load_model(MODEL_NAME)
        encoder_model = Model(encoder_model.inputs, encoder_model.layers[-2].output)
        # Should probably do something to identify the non-RP weights here
        # Freeze Layers
        for layer in encoder_model.layers:
            layer.name += "_encoder"
            layer.trainable = False
        actor = Dense(512)(encoder_model.outputs[0])
        actor = Activation('relu')(actor)
        actor = Dense(24)(actor)
        actor = Activation('relu')(actor)
        actor = Dense(self.action_size)(actor)
        actor = Activation('softmax')(actor)
        model = Model(inputs=encoder_model.inputs, outputs=actor)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.actor_lr))
        model.summary()
        return model

    def build_actor(self):
        # Inputs
        input1 = Input(shape=(64, 64, 3), name="input1_actor")
        input2 = Input(shape=(64, 64, 3), name="input2_actor")
        input3 = Input(shape=(64, 64, 3), name="input3_actor")


        # Conv 1
        conv_layer1 = Conv2D(6, 3);
        batchnorm_layer1 = BatchNormalization()
        maxpool_layer1 = MaxPool2D()

        conv1_input_1 = maxpool_layer1(batchnorm_layer1(conv_layer1(input1)))
        conv1_input_2 = maxpool_layer1(batchnorm_layer1(conv_layer1(input2)))
        conv1_input_3 = maxpool_layer1(batchnorm_layer1(conv_layer1(input3)))

        # Conv 2
        conv_layer2 = Conv2D(12, 3);
        batchnorm_layer2 = BatchNormalization()
        maxpool_layer2 = MaxPool2D()

        conv2_input_1 = maxpool_layer2(batchnorm_layer2(conv_layer2(conv1_input_1)))
        conv2_input_2 = maxpool_layer2(batchnorm_layer2(conv_layer2(conv1_input_2)))
        conv2_input_3 = maxpool_layer2(batchnorm_layer2(conv_layer2(conv1_input_3)))

        # Conv 3
        conv_layer3 = Conv2D(12, 3);
        batchnorm_layer3 = BatchNormalization()
        maxpool_layer3 = MaxPool2D()

        conv3_input_1 = maxpool_layer3(batchnorm_layer3(conv_layer3(conv2_input_1)))
        conv3_input_2 = maxpool_layer3(batchnorm_layer3(conv_layer3(conv2_input_2)))
        conv3_input_3 = maxpool_layer3(batchnorm_layer3(conv_layer3(conv2_input_3)))

        # Conv 4
        # conv_layer4 = Conv2D(24, 3);
        # batchnorm_layer4 = BatchNormalization()
        # maxpool_layer4 = MaxPool2D()

        # conv4_input_1 = maxpool_layer4(batchnorm_layer4(conv_layer4(conv3_input_1)))
        # conv4_input_2 = maxpool_layer4(batchnorm_layer4(conv_layer4(conv3_input_2)))
        # conv4_input_3 = maxpool_layer4(batchnorm_layer4(conv_layer4(conv3_input_3)))

        # Flatten
        flattened_input_1 = Flatten()(conv3_input_1)
        flattened_input_2 = Flatten()(conv3_input_2)
        flattened_input_3 = Flatten()(conv3_input_3)

        # Concatenate
        concatenated = concatenate([flattened_input_1, flattened_input_2, flattened_input_3])

        output = concatenated
        output = Dense(512, activation='relu', kernel_initializer='he_uniform')(output)
        output = Dense(256, activation='relu', kernel_initializer='he_uniform')(output)
        output = Dense(128, activation='relu', kernel_initializer='he_uniform')(output)
        output = Dense(24, activation='relu', kernel_initializer='he_uniform')(output)
        output = Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform')(output)

        actor = Model(inputs = [input1, input2, input3], outputs=[output])

        actor.summary()

        # See note regarding crossentropy in cartpole_reinforce.py
        actor.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.actor_lr))
        return actor

    # critic: state is input and value of state is output of model
    def build_critic_thesis(self):
        # Get the RP extrcritic pre-trained model
        encoder_model = load_model(MODEL_NAME)
        encoder_model = Model(encoder_model.inputs, encoder_model.layers[-2].output)
        # Should probably do something to identify the non-RP weights here
        # Freeze Layers
        for layer in encoder_model.layers:
            layer.name += "_encoder"
            layer.trainable = False
        critic = Dense(512)(encoder_model.outputs[0])
        critic = Activation('relu')(critic)
        critic = Dense(24)(critic)
        critic = Activation('relu')(critic)
        critic = Dense(self.value_size)(critic)
        critic = Activation('linear')(critic)
        model = Model(inputs=encoder_model.inputs, outputs=critic)
        model.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
        model.summary()
        return model

    def build_critic(self):
        # Inputs
        input1 = Input(shape=(64, 64, 3), name="input1_critic")
        input2 = Input(shape=(64, 64, 3), name="input2_critic")
        input3 = Input(shape=(64, 64, 3), name="input3_critic")


        # Conv 1
        conv_layer1 = Conv2D(6, 3);
        batchnorm_layer1 = BatchNormalization()
        maxpool_layer1 = MaxPool2D()

        conv1_input_1 = maxpool_layer1(batchnorm_layer1(conv_layer1(input1)))
        conv1_input_2 = maxpool_layer1(batchnorm_layer1(conv_layer1(input2)))
        conv1_input_3 = maxpool_layer1(batchnorm_layer1(conv_layer1(input3)))

        # Conv 2
        conv_layer2 = Conv2D(12, 3);
        batchnorm_layer2 = BatchNormalization()
        maxpool_layer2 = MaxPool2D()

        conv2_input_1 = maxpool_layer2(batchnorm_layer2(conv_layer2(conv1_input_1)))
        conv2_input_2 = maxpool_layer2(batchnorm_layer2(conv_layer2(conv1_input_2)))
        conv2_input_3 = maxpool_layer2(batchnorm_layer2(conv_layer2(conv1_input_3)))

        # Conv 3
        conv_layer3 = Conv2D(12, 3);
        batchnorm_layer3 = BatchNormalization()
        maxpool_layer3 = MaxPool2D()

        conv3_input_1 = maxpool_layer3(batchnorm_layer3(conv_layer3(conv2_input_1)))
        conv3_input_2 = maxpool_layer3(batchnorm_layer3(conv_layer3(conv2_input_2)))
        conv3_input_3 = maxpool_layer3(batchnorm_layer3(conv_layer3(conv2_input_3)))

        # Conv 4
        # conv_layer4 = Conv2D(24, 3);
        # batchnorm_layer4 = BatchNormalization()
        # maxpool_layer4 = MaxPool2D()

        # conv4_input_1 = maxpool_layer4(batchnorm_layer4(conv_layer4(conv3_input_1)))
        # conv4_input_2 = maxpool_layer4(batchnorm_layer4(conv_layer4(conv3_input_2)))
        # conv4_input_3 = maxpool_layer4(batchnorm_layer4(conv_layer4(conv3_input_3)))

        # Flatten
        flattened_input_1 = Flatten()(conv3_input_1)
        flattened_input_2 = Flatten()(conv3_input_2)
        flattened_input_3 = Flatten()(conv3_input_3)

        # Concatenate
        concatenated = concatenate([flattened_input_1, flattened_input_2, flattened_input_3])

        output = concatenated
        output = Dense(512, activation='relu', kernel_initializer='he_uniform')(output)
        output = Dense(256, activation='relu', kernel_initializer='he_uniform')(output)
        output = Dense(128, activation='relu', kernel_initializer='he_uniform')(output)
        output = Dense(24, activation='relu', kernel_initializer='he_uniform')(output)
        output = Dense(self.value_size, activation='linear', kernel_initializer='he_uniform')(output)

        critic = Model(inputs = [input1, input2, input3], outputs=[output])

        critic.summary()
        critic.compile(loss="mse", optimizer=Adam(lr=self.critic_lr))
        return critic

    # using the output of policy network, pick action stochastically
    def get_action(self, states_history):
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
    # In case of CartPole-v1, maximum length of episode is 500
    env = gym.make('cartpole-visual-v1')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # make A2C agent
    agent = A2CAgent(state_size, action_size)

    scores, episodes = [], []

    with open('save_graph/visual_baseline_history_more_dense3.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for e in trange(EPISODES):
            states_history = []

            done = False
            score = 0
            state = env.reset()
            state = np.reshape(state, (1,) + state.shape)


            i = 0
            while not done:
                if agent.render:
                    env.render()

                if len(states_history) >= 3:
                    states_history.pop(0)
                    states_history.append(state)
                    action = agent.get_action(states_history)
                    next_state, reward, done, info = env.step(action)


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
