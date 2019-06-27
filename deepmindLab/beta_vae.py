#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose, Concatenate
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy, mean_absolute_error, mae
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence


import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
import cv2
import sys
from tqdm import tqdm, trange


input_shape=(84, 84, 3)
latent_dim = 100
inputs = Input(shape=input_shape, name='denoising_encoder_input')
x_inputs = Conv2D(filters=32, kernel_size=4, activation='relu', strides=2, padding='same')(inputs)
x_inputs = Conv2D(filters=32, kernel_size=4, activation='relu', strides=2, padding='same')(x_inputs)
x_inputs = Conv2D(filters=64, kernel_size=4, activation='relu', strides=2, padding='same')(x_inputs)
x_inputs = Conv2D(filters=64, kernel_size=4, activation='relu', strides=2, padding='same')(x_inputs)

x_inputs = Flatten()(x_inputs)
z = Dense(latent_dim, name='z', activation='linear')(x_inputs)

# instantiate encoder model
encoder = Model(inputs, z, name='denoising_encoder')

# build decoder model
latent_inputs = Input(shape=(latent_dim,))
x_decoder = Dense(6 * 6 * 64, activation='relu')(latent_inputs)
x_decoder = Reshape((6, 6, 64))(x_decoder)

x_decoder = Conv2DTranspose(filters=64, kernel_size=4, activation='relu', strides=2, padding='same')(x_decoder)
x_decoder = Conv2DTranspose(filters=64, kernel_size=4, activation='relu', strides=2, padding='same')(x_decoder)
x_decoder = Conv2DTranspose(filters=32, kernel_size=4, activation='relu', strides=2, padding='same')(x_decoder)
x_decoder = Conv2DTranspose(filters=32, kernel_size=4, activation='relu', strides=2, padding='same')(x_decoder)

x_decoder = Conv2DTranspose(filters=3, kernel_size=1, strides=1, activation='linear', padding='same')(x_decoder)
x_decoder = Lambda(lambda x: x[:, :84, :84, :])(x_decoder)

# instantiate decoder model
decoder = Model(latent_inputs, x_decoder, name='denoising_decoder')

# instantiate VAE model
denoising = Model(inputs, decoder(encoder(inputs)), name='denoising')
denoising.load_weights("denoising_autoencoder.h5")

for layer in denoising.layers:
    layer.name += "_denoising"


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, 84, 84, 3))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def sampling_np(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = np.shape(z_mean)[0]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = np.random_normal(shape=(batch, 84, 84, 3))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# network parameters
latent_dim = 32
input_shape = (84, 84, 3)

# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x_inputs = Conv2D(filters=32, kernel_size=4, activation='relu', strides=2, padding='same')(inputs)
x_inputs = Conv2D(filters=32, kernel_size=4, activation='relu', strides=2, padding='same')(x_inputs)
x_inputs = Conv2D(filters=64, kernel_size=4, activation='relu', strides=2, padding='same')(x_inputs)
x_inputs = Conv2D(filters=64, kernel_size=4, activation='relu', strides=2, padding='same')(x_inputs)

x_inputs = Flatten()(x_inputs)
x_inputs = Dense(256, activation='relu')(x_inputs)
z_mean = Dense(latent_dim, name='z_mean', activation='linear')(x_inputs)
z_log_var = Dense(latent_dim, name='z_log_var', activation='linear')(x_inputs)

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var], name='encoder')
encoder.summary()


# build decoder model
input_z_mean = Input(shape=(latent_dim,))
input_z_log_var = Input(shape=(latent_dim,))
latent_inputs = Concatenate()([input_z_mean, input_z_log_var])
x_decoder = Dense(256, activation='relu')(latent_inputs)
x_decoder = Dense(6 * 6 * 64, activation='relu')(x_decoder)
x_decoder = Reshape((6, 6, 64))(x_decoder)

x_decoder = Conv2DTranspose(filters=64, kernel_size=4, activation='relu', strides=2, padding='same')(x_decoder)
x_decoder = Conv2DTranspose(filters=64, kernel_size=4, activation='relu', strides=2, padding='same')(x_decoder)
x_decoder = Conv2DTranspose(filters=32, kernel_size=4, activation='relu', strides=2, padding='same')(x_decoder)
x_decoder = Conv2DTranspose(filters=32, kernel_size=4, activation='relu', strides=2, padding='same')(x_decoder)

x_decoder = Conv2DTranspose(filters=6, kernel_size=1, strides=1, activation='linear', padding='same')(x_decoder)
x_decoder = Lambda(lambda x: x[:, :84, :84, :])(x_decoder)

# instantiate decoder model
decoder = Model([input_z_mean, input_z_log_var], x_decoder, name='decoder')
decoder.summary()

# instantiate VAE model
encoder_outputs = encoder(inputs)
outputs = decoder([encoder_outputs[0], encoder_outputs[1]])
vae = Model(inputs, outputs, name='vae')
for layer in vae.layers:
    layer.name += "_vae"


denoising_encoder = Model(denoising.inputs, denoising.layers[-2].outputs)
for layer in denoising_encoder.layers:
    layer.trainable = False


def load_small_dataset():
    imgs = np.asarray([cv2.imread(x) for x in tqdm(glob.glob("training_observations2/*.png")[:300])])
    print(imgs.shape)
    x_train = imgs[:128, :]
    x_test = imgs[128:, :]
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    y_train = np.array([])
    y_test = np.array([])
    image_size = imgs.shape[1]
    input_shape = (image_size, image_size, 3)

    return x_train, y_train, x_test, y_test, image_size, input_shape

x_train, y_train, x_test, y_test, image_size, input_shape = load_small_dataset()


output = vae.outputs[0]
mean_output = output[:, :, :, :3]
log_var_output = output[:, :, :, 3:]
sampled_reconstruction = sampling([mean_output, log_var_output])
reconstruction_loss = K.square(denoising_encoder(inputs) - denoising_encoder(sampled_reconstruction))
reconstruction_loss = K.mean(reconstruction_loss, axis=-1)
beta = 1.
kl_loss = 1 + mean_output - K.square(log_var_output) - K.exp(mean_output)
kl_loss = K.mean(kl_loss, axis=[-1, -2, -3])
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + beta * kl_loss)
vae.add_loss(vae_loss)
learning_rate = 1e-4
adam = Adam(lr=learning_rate)
vae.compile(optimizer=adam)
vae.summary()


class DataSequence(Sequence):
    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        self.filenames = glob.glob("training_observations2/*.png")
        self.image_size = 84
        self.on_epoch_end()

    def on_epoch_end(self):
        np.random.shuffle(self.filenames)

    def __len__(self):
        return int(np.floor(len(self.filenames) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = np.asarray([cv2.imread(x) for x in self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]])
        batch_x = batch_x.astype('float32') / 255.
        return batch_x, None



epochs = 1
batch_size = 100
steps_per_epoch = int(np.round(len(glob.glob("training_observations2/*.png")) / batch_size))
checkpoint = ModelCheckpoint('beta_vae_checkpoint.h5', monitor='val_loss', verbose=0, save_best_only=True, mode='min', save_weights_only=True)

if True:
    img_generator = DataSequence(batch_size=batch_size)
    vae.load_weights('darla_vae.h5')
    history = vae.fit_generator(img_generator, epochs=epochs, workers=7, callbacks=[checkpoint], validation_data=(x_test, None))


    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.legend(['train', 'test'])
    # plt.show()


    vae.save_weights('darla_vae.h5')
else:
    vae.load_weights('darla_vae.h5')


predicted_imgs = vae.predict(x_train, batch_size=batch_size)[:, :, :, :3]
print(x_train.max(), x_train.mean(), x_train.min())
cv2.imwrite("original.png", x_train[35] * 255.)

if False:
    print(predicted_imgs.max(), predicted_imgs.mean(), predicted_imgs.min())
    predicted_imgs = (predicted_imgs - predicted_imgs.min()) / (predicted_imgs.max() - predicted_imgs.min())
    print(predicted_imgs.max(), predicted_imgs.mean(), predicted_imgs.min())
    cv2.imwrite("predicted.png", predicted_imgs[118])
else:
    print(predicted_imgs.shape)
    denoised_predicted = denoising.predict(predicted_imgs)
    print(denoised_predicted.max(), denoised_predicted.mean(), denoised_predicted.min())
    denoised_predicted = np.clip(denoised_predicted, 0., 1.)
    cv2.imwrite("denoised.png", denoised_predicted[35] * 255.)
