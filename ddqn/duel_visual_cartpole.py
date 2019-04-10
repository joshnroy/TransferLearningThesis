import numpy as np
import gym
import gym_cartpole_visual

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Conv2D, BatchNormalization, MaxPool2D, Reshape, Lambda, concatenate
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import sys
from pyvirtualdisplay import Display

display = Display(visible=0, size=(100, 100))
display.start()


ENV_NAME = 'cartpole-visual-v1'
NUM_FILTERS = 6
NUM_CONV_LAYERS = 3
NUM_HIDDEN_LAYERS = 3
HIDDEN_LAYER_SIZE = 48


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model regardless of the dueling architecture
# if you enable dueling network in DQN , DQN will build a dueling network base on your model automatically
# Also, you can build a dueling network by yourself and turn off the dueling network in DQN.
# print(env.observation_space.shape)
# sys.exit()
inputs = Input(shape=(1,) + (64 * 64 * 3 + 2,), name="meme_input")
# Convolutional Layers
flat_img_input = Lambda(lambda x: x[:, :, :-2])(inputs)
vel_input = Lambda(lambda x: x[:, :, -2:])(inputs)
conv = Reshape(target_shape=(64, 64, 3))(flat_img_input)
vel_input = Reshape(target_shape=(2,))(vel_input)
# vel_input = inputs[:, :, -2:2]
numFilters = NUM_FILTERS
for i in range(NUM_CONV_LAYERS):
    conv_layer = Conv2D(numFilters, 3)
    batchnorm_layer = BatchNormalization()
    maxpool_layer = MaxPool2D()
    conv = maxpool_layer(batchnorm_layer(conv_layer(conv)))
    numFilters *= 2
outputs = Flatten()(conv)
for _ in range(NUM_HIDDEN_LAYERS-1):
    outputs = Dense(HIDDEN_LAYER_SIZE, activation='relu')(outputs)
outputs = concatenate([outputs, vel_input])
outputs = Dense(HIDDEN_LAYER_SIZE, activation='relu')(outputs)
outputs = Dense(nb_actions, activation='linear')(outputs)
model = Model(inputs, outputs)
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
# enable the dueling network
# you can specify the dueling_type to one of {'avg','max','naive'}
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-5), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

# After training is done, we save the final weights.
dqn.save_weights('duel_dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=False)
