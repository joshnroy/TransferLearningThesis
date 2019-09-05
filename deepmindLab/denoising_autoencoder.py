from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy, mean_absolute_error
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K
from keras.utils import Sequence
import keras

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
import cv2
import sys
import random

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

image_size = 84
input_shape = (image_size, image_size, 3)

# network parameters
batch_size = 128
latent_dim = 100
epochs = 3

def remove_patch(x):
    low_x = np.random.randint(0, image_size-1)
    high_x = np.random.randint(low_x, image_size)
    low_y = np.random.randint(0, image_size-1)
    high_y = np.random.randint(low_y, image_size)

    x = np.copy(x)
    x[low_x:high_x, low_y:high_y, :] = 0

    return x

class DataSequence(Sequence):
    def __init__(self):
        self.filenames = glob.glob("vae_training_data/*")
        self.image_size = 84
        self.curr_episode = 0
        self.i = 0

    def on_epoch_end(self):
        pass

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        batch_y = np.load(self.filenames[self.i])['observations']
        self.i = (self.i + 1) // len(self.filenames)
        batch_x = np.array([remove_patch(np.copy(y)) for y in batch_y])

        return batch_x, batch_y

# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x_inputs = Conv2D(filters=32, kernel_size=4, activation='relu', strides=2,
                  padding='same')(inputs)
x_inputs = Conv2D(filters=32, kernel_size=4, activation='relu', strides=2,
                  padding='same')(x_inputs)
x_inputs = Conv2D(filters=64, kernel_size=4, activation='relu', strides=2,
                  padding='same')(x_inputs)
x_inputs = Conv2D(filters=64, kernel_size=4, activation='relu', strides=2,
                  padding='same')(x_inputs)

x_inputs = Flatten()(x_inputs)
x_inputs = Dense(256, activation='relu')(x_inputs)
z = Dense(latent_dim, name='z', activation='relu')(x_inputs)

# build decoder model
x_decoder = Dense(256, activation='relu')(z)
x_decoder = Dense(6 * 6 * 64, activation='relu')(x_decoder)
x_decoder = Reshape((6, 6, 64))(x_decoder)

x_decoder = Conv2DTranspose(filters=64, kernel_size=4, activation='relu',
                            strides=2, padding='same')(x_decoder)
x_decoder = Conv2DTranspose(filters=64, kernel_size=4, activation='relu',
                            strides=2, padding='same')(x_decoder)
x_decoder = Conv2DTranspose(filters=32, kernel_size=4, activation='relu',
                            strides=2, padding='same')(x_decoder)
x_decoder = Conv2DTranspose(filters=32, kernel_size=4, activation='relu',
                            strides=2, padding='same')(x_decoder)

x_decoder = Conv2DTranspose(filters=3, kernel_size=1, strides=1,
                            activation='sigmoid', padding='same')(x_decoder)
x_decoder = Lambda(lambda x: x[:, :84, :84, :])(x_decoder)

# instantiate
denoising = Model(inputs, x_decoder, name='vae')
denoising.compile(loss='mse', optimizer='adam')

if __name__ == '__main__':
    img_generator = DataSequence()
    denoising.load_weights("denoising_autoencoder.h5")
    history = denoising.fit_generator(img_generator, epochs=epochs, workers=9)
    denoising.save_weights("denoising_autoencoder.h5")
    denoising.save("full_denoising_autoencoder.h5")

    # Test the autoencoder
    if True:
        y_test = np.load(glob.glob("vae_training_data/*.npz")[0])['observations']
        x_test = np.array([remove_patch(np.copy(y)) for y in y_test])

        predicted_imgs = denoising.predict(x_test, batch_size=batch_size)
        print("test")
        print(x_test.max(), x_test.min(), x_test.mean())
        print(predicted_imgs.max(), predicted_imgs.min(), predicted_imgs.mean())
        x_test = x_test * 255.
        predicted_imgs = predicted_imgs * 255.
        print(x_test.max(), x_test.min(), x_test.mean())
        print(predicted_imgs.max(), predicted_imgs.min(), predicted_imgs.mean())
        for i in range(len(predicted_imgs)):
            cv2.imwrite("test_images/original" + str(i) + ".png", x_test[i])
            cv2.imwrite("test_images/reconstructed" + str(i) + ".png", predicted_imgs[i])
