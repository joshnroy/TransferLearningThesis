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

import matplotlib.pyplot as plt

from copy import deepcopy

from pyvirtualdisplay import Display

display = Display(visible=0, size=(100, 100))
display.start()

NUM_FILTERS = 6
NUM_CONV_LAYERS = 3
NUM_HIDDEN_LAYERS = 10
HIDDEN_LAYER_SIZE = 48


if __name__ == "__main__":
    loaded_data = np.load("pos_regressor_data.npz")
    input_imgs = loaded_data["observations"]
    true_states = loaded_data["states"]
    true_states = true_states * 100

    # Inputs
    img_input = Input(shape=(64, 64, 3), name="input_img")
    vel_input = Input(shape=(2,), name="input_vel")
    inputs = [img_input, vel_input]

    # Convolutional Layers
    conv = img_input
    numFilters = NUM_FILTERS
    for i in range(NUM_CONV_LAYERS):
        conv_layer = Conv2D(numFilters, 3)
        batchnorm_layer = BatchNormalization()
        maxpool_layer = MaxPool2D()
        conv = maxpool_layer(batchnorm_layer(conv_layer(conv)))
        numFilters *= 2

    # Flatten
    flattened = Flatten()(conv)

    for _ in range(NUM_HIDDEN_LAYERS-1):
        flattened = Dense(HIDDEN_LAYER_SIZE, activation='relu')(flattened)
    output = concatenate([flattened, vel_input])
    output = Dense(HIDDEN_LAYER_SIZE, activation='relu')(output)
    output = Dense(4, activation='linear')(output)

    model = Model(inputs = inputs, outputs=[output])

    learning_rate = 1e-3
    num_epochs = 750
    decay = learning_rate / float(num_epochs)

    model.compile(loss='mean_absolute_error',
                  optimizer=Adam(lr=learning_rate, decay=decay))
    model.summary()

    velocities = np.asarray([true_states[:, 1], true_states[:, 3]])
    velocities = np.transpose(velocities)
    hist = model.fit([input_imgs, velocities], true_states, batch_size=100, epochs=num_epochs, validation_split=0.2)
    for i in range(true_states[0:10].shape[0]):
        print(model.predict([np.asarray([input_imgs[i]]), np.asarray([velocities[i]])]), "\t\t", true_states[i])
    # Plot training & validation loss values
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("pos_regressor_training2.png")
