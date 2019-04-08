import numpy as np
import gym
import gym_cartpole_visual

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Conv2D, BatchNormalization, MaxPool2D, Reshape
from keras.optimizers import Adam

from variational_autoencoder_deconv import vae

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import sys
from pyvirtualdisplay import Display

display = Display(visible=0, size=(100, 100))
display.start()


ENV_NAME = 'cartpole-visual-v1'
NUM_FILTERS = 3
NUM_CONV_LAYERS = 3
NUM_HIDDEN_LAYERS = 3
HIDDEN_LAYER_SIZE = 48

keras.losses.custom_loss = keras.losses.mean_squared_error
MODEL_FILE = "../../encoder_results/baseline_encoder"


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model regardless of the dueling architecture
# if you enable dueling network in DQN , DQN will build a dueling network base on your model automatically
# Also, you can build a dueling network by yourself and turn off the dueling network in DQN.
stylegan_encoder = keras.models.load_model(MODEL_FILE)
# vae_model = Model(vae.inputs, [vae.layers[-2].outputs[2]])
for layer in stylegan_encoder.layers:
    layer.trainable = False
# print(env.observation_space.shape)
# sys.exit()
inputs = Input(shape=(1,) + env.observation_space.shape, name="meme_input")
# Convolutional Layers
conv = Reshape(target_shape=(64, 64, 3))(inputs)
outputs = stylegan_encoder(conv)
for _ in range(NUM_HIDDEN_LAYERS):
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
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-1), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
dqn.fit(env, nb_steps=500000, visualize=False, verbose=1)

# After training is done, we save the final weights.
dqn.save_weights('duel_dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=False)
