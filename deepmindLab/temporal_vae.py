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
from keras.callbacks import ModelCheckpoint, Callback
from keras.utils import Sequence


import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
import cv2
import sys
from tqdm import tqdm, trange

rp_l2_loss_weight = 0.5
s_entropy_loss_term = 0.5


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
rp_dim = int(np.round(latent_dim * 0.5))
s_dim = latent_dim - rp_dim
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
outputs = [decoder([encoder_outputs[0], encoder_outputs[1]]), encoder_outputs[0], encoder_outputs[1]]
vae = Model(inputs, outputs, name='vae')
for layer in vae.layers:
    layer.name += "_vae"


denoising_encoder = Model(denoising.inputs, denoising.layers[-2].outputs)
for layer in denoising_encoder.layers:
    layer.trainable = False


def load_small_dataset():
    imgs = np.asarray([cv2.imread(x) for x in tqdm(glob.glob("training_observations2/obs_104_*.png") + glob.glob("training_observations2/obs_105_*.png"))])
    x_train = imgs[:, :]
    x_test = imgs[128:, :]
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    y_train = np.array([])
    y_test = np.array([])
    image_size = imgs.shape[1]
    input_shape = (image_size, image_size, 3)

    return x_train, y_train, x_test, y_test, image_size, input_shape

x_train, y_train, x_test, y_test, image_size, input_shape = load_small_dataset()

def recon_loss(y_true, y_pred):
    output = y_pred[0]
    mean_output = output[:, :, :, :3]
    log_var_output = output[:, :, :, 3:]
    sampled_reconstruction = sampling([mean_output, log_var_output])
    reconstruction_loss = K.square(denoising_encoder(inputs) - denoising_encoder(sampled_reconstruction))
    reconstruction_loss = K.mean(reconstruction_loss, axis=-1)
    return reconstruction_loss

def rp_loss(y_true, y_pred):
    mean_latent = y_pred[1]
    log_var_latent = y_pred[2]
    rp_l2_loss = rp_l2_loss_weight * (K.mean(K.square(mean_latent[1:360, :rp_dim] -
                                                      mean_latent[0:359, :rp_dim])) +
                                      K.mean(K.square(log_var_latent[1:360, :rp_dim] -
                                                      log_var_latent[0:359, :rp_dim])))

    rp_l2_loss += rp_l2_loss_weight * (K.mean(K.square(mean_latent[361:720, :rp_dim] -
                                                      mean_latent[360:719, :rp_dim])) +
                                      K.mean(K.square(log_var_latent[361:720, :rp_dim] -
                                                      log_var_latent[360:719, :rp_dim])))
    return rp_l2_loss


def entropy(x):
    return 0.5

def entropy_loss(y_true, y_pred):
    mean_latent = y_pred[1]
    log_var_latent = y_pred[2]
    # s_entropy_loss = s_entropy_loss_term * (entropy(mean_latent) + entropy(log_var_latent))
    s_entropy_loss = s_entropy_loss_term * 0.5 * K.max([K.mean(K.sum(log_var_latent[:, rp_dim:], axis=1)), 0.])
    return s_entropy_loss

def s_loss(y_true, y_pred):
    mean_latent = y_pred[1]
    log_var_latent = y_pred[2]

    s_l2_loss = s_entropy_loss_term * K.mean(K.square(K.mean(mean_latent[0:360, rp_dim:], axis=0) - K.mean(mean_latent[360:720, rp_dim:], axis=0)))
    s_l2_loss += s_entropy_loss_term * K.mean(K.square(K.mean(log_var_latent[:360, rp_dim:], axis=0) - K.mean(log_var_latent[360:, rp_dim:], axis=0)))

    return s_l2_loss



output = vae.outputs[0]
mean_output = output[:, :, :, :3]
log_var_output = output[:, :, :, 3:]
reconstruction_loss = recon_loss(None, vae.outputs)
rp_l2_loss = rp_loss(None, vae.outputs)
# s_entropy_loss = entropy_loss(None, vae.outputs)
s_l2_loss = s_loss(None, vae.outputs)


beta = 1.
kl_loss = 1 + mean_output - K.square(log_var_output) - K.exp(mean_output)
kl_loss = K.mean(kl_loss, axis=[-1, -2, -3])
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + beta * kl_loss) + K.mean(rp_l2_loss) + K.mean(s_l2_loss)
vae.add_loss(vae_loss)

learning_rate = 1e-4
adam = Adam(lr=learning_rate)
vae.compile(optimizer=adam)
vae.summary()


class DataSequence(Sequence):
    def __init__(self):
        self.num_episodes = 2500
        self.num_frames = 360
        self.filenames = [["training_observations2/obs_" + str(i) +  "_" + str(j) + ".png" for j in range(self.num_frames)] for i in range(self.num_episodes)]
        self.image_size = 84
        self.curr_episode = 0
        # self.on_epoch_end()

    def on_epoch_end(self):
        pass

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        batch_x = []
        while len(batch_x) != 720:
            batch_x = np.asarray([cv2.imread(x) for x in self.filenames[self.curr_episode]])
            batch_x = []
            for f in self.filenames[self.curr_episode]:
                x = cv2.imread(f)
                if x is not None:
                    batch_x.append(x)
            for f in self.filenames[self.curr_episode+1]:
                x = cv2.imread(f)
                if x is not None:
                    batch_x.append(x)
            batch_x = np.asarray(batch_x)
            batch_x = batch_x.astype('float32') / 255.
            self.curr_episode = (self.curr_episode + 2) % self.num_episodes
        return batch_x, None



epochs = 10 #30, 2
checkpoint = ModelCheckpoint('temporal_vae_checkpoint.h5', monitor='loss', verbose=0, save_best_only=True, mode='min', save_weights_only=True)

if False:
    vae.load_weights('temporal_vae4.h5')
    img_generator = DataSequence()
    history = vae.fit_generator(img_generator, epochs=epochs, validation_data=(x_train, None))
    vae.save_weights('temporal_vae4.h5')


    # plt.plot(history.history['loss'], label='train')
    # plt.plot(history.history['val_loss'], label='test')
    # plt.legend(['train', 'test'])
    # plt.show()


else:
    vae.load_weights('temporal_vae4.h5')


predicted_outputs = vae.predict(x_train)
predicted_imgs = predicted_outputs[0][:, :, :, :3]
cv2.imwrite("original.png", x_train[35] * 255.)

predicted_means = predicted_outputs[1]
predicted_log_vars = predicted_outputs[2]
predicted_rp_l2 = np.mean((predicted_means[1:, :rp_dim] - predicted_means[:-1, :rp_dim])**2 + (predicted_log_vars[1:, :rp_dim] - predicted_log_vars[:-1, :rp_dim]))

denoised_originals = denoising_encoder.predict(x_train)
denoised_predicted = denoising_encoder.predict(predicted_imgs)
predicted_reconstruction_loss = np.mean((denoised_originals - denoised_predicted)**2)

predicted_s_loss = np.mean((np.mean(predicted_outputs[1][0:360, rp_dim:], axis=0) - np.mean(predicted_outputs[1][360:720, rp_dim:], axis=0)) ** 2)
predicted_s_loss += np.mean((np.mean(predicted_outputs[2][:360, rp_dim:], axis=0) - np.mean(predicted_outputs[2][360:, rp_dim:], axis=0)) ** 2)

# print(predicted_means[0, :], predicted_log_vars[0, :])
# plt.plot(predicted_means[0, :], color='r')
# plt.plot(predicted_log_vars[0, :], color='b')
# plt.show()

# sys.exit()
print("RP L2 LOSS", predicted_rp_l2)
print("S LOSS", predicted_s_loss)
# for i in range(16):
#     plt.plot(predicted_means[:, i], color='r', alpha=1)
#     # plt.plot(predicted_log_vars[:, i], color='r', alpha=1)
# for i in range(16, 32):
#     plt.plot(predicted_means[:, i], color='b', alpha=0.2)
#     # plt.plot(predicted_log_vars[:, i], color='b', alpha=0.2)
# plt.show()
print("RECONSTRUCTION LOSS", predicted_reconstruction_loss)

if False:
    predicted_imgs = (predicted_imgs - predicted_imgs.min()) / (predicted_imgs.max() - predicted_imgs.min())
    cv2.imwrite("predicted.png", predicted_imgs[118])
else:
    denoised_predicted = denoising.predict(predicted_imgs)
    denoised_imgs = np.clip(denoised_predicted[35], 0., 1.)
    cv2.imwrite("denoised_temporal.png", denoised_imgs * 255.)
