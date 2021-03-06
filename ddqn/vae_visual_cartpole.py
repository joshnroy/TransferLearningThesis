import numpy as np
import gym
import gym_cartpole_visual

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Conv2D, BatchNormalization, MaxPool2D, Reshape, Lambda, concatenate, Dropout
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import sys
from pyvirtualdisplay import Display

from variational_autoencoder_deconv import vae

import matplotlib.pyplot as plt

display = Display(visible=0, size=(100, 100))
display.start()

WEIGHTS_FILE = "vae_cnn_cartpole.h5"

ENV_NAME = 'cartpole-visual-v1'
NUM_FILTERS = 6
NUM_CONV_LAYERS = 3
NUM_HIDDEN_LAYERS = 5
HIDDEN_LAYER_SIZE = 48


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# vae.load_weights(WEIGHTS_FILE)

vae = Model(vae.inputs, [vae.layers[-2].outputs[2]])
for layer in vae.layers:
    layer.trainable = False

# Next, we build a very simple model regardless of the dueling architecture
# if you enable dueling network in DQN , DQN will build a dueling network base on your model automatically
# Also, you can build a dueling network by yourself and turn off the dueling network in DQN.
# print(env.observation_space.shape)
# sys.exit()
inputs = Input(shape=(1,) + (64 * 64 * 3 + 2,), name="meme_input")
flat_img_input = Lambda(lambda x: x[:, :, :-2])(inputs)
vel_input = Lambda(lambda x: x[:, :, -2:])(inputs)
conv = Reshape(target_shape=(64, 64, 3))(flat_img_input)
vel_input = Reshape(target_shape=(2,))(vel_input)

# Convolutional Layers
outputs = vae(conv)
for _ in range(NUM_HIDDEN_LAYERS-1):
    outputs = Dense(HIDDEN_LAYER_SIZE, activation='relu')(outputs)
outputs = concatenate([outputs, vel_input])
outputs = Dense(HIDDEN_LAYER_SIZE, activation='relu')(outputs)
outputs = Dense(nb_actions, activation='linear')(outputs)
model = Model(inputs, outputs)
# print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
# enable the dueling network
# you can specify the dueling_type to one of {'avg','max','naive'}
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
# history = dqn.fit(env, nb_steps=500000, visualize=False, verbose=1)
# np.savez_compressed("vae_dqn_training_history_500k", episode_reward=np.asarray(history.history['episode_reward']))

# After training is done, we save the final weights.
dqn.load_weights('vae_dqn_{}_weights.h5f'.format(ENV_NAME))

# Finally, evaluate our algorithm for 5 episodes.
# print("Test on original colors")
dqn.test(env, nb_episodes=10, visualize=False)

# print("Test on changed colors")
env.change_color()
dqn.test(env, nb_episodes=10, visualize=False)
