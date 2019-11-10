from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose, Concatenate
from keras.models import Model, load_model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy, mean_absolute_error, mae
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback
from keras.utils import Sequence
import keras

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
import cv2
import sys
from tqdm import tqdm, trange

from skimage import io


from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


epochs = 3

class DataSequence(Sequence):
    def __init__(self):
        self.filenames = glob.glob("training_data_small2/*.npz")
        self.image_size = 32
        self.i = 0
        self.batch_size = 50
        self.npz_idx = 0
        self.data = np.load(self.filenames[self.npz_idx])
        self.images = self.data["images"]
        self.npz_len = 10000

    def __len__(self):
        return len(self.filenames) * self.npz_len // self.batch_size

    def __getitem__(self, idx):
        if (self.i * self.batch_size) > self.npz_len:
            self.i = 0
            self.npz_idx = (self.npz_idx + 1) // len(self.filenames)
            self.data = np.load(self.filenames[self.npz_idx])
            self.images = self.data["images"]
        self.i += 1

        # for j, img in enumerate(self.images[self.i:self.i+self.batch_size, :, :, :]):
        #     io.imsave("test" + str(j) + ".png", img)

        batch_x = self.images[self.i:self.i+self.batch_size, :, :, :]

        return batch_x, None

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
    epsilon = K.random_normal(shape=(batch, 32, 32, 3))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# network parameters
latent_dim = 32
rp_dim = int(np.round(latent_dim * 0.5))
s_dim = latent_dim - rp_dim
input_shape = (32, 32, 3)

# build encoder network
encoder_input = Input(shape=input_shape, name='encoder_input')
x_inputs = Conv2D(filters=32, kernel_size=4, activation='relu', strides=2, padding='same')(encoder_input)
x_inputs = Conv2D(filters=32, kernel_size=4, activation='relu', strides=2, padding='same')(x_inputs)
x_inputs = Conv2D(filters=64, kernel_size=4, activation='relu', strides=2, padding='same')(x_inputs)

x_inputs = Flatten()(x_inputs)
x_inputs = Dense(256, activation='relu')(x_inputs)

z_mean = Dense(latent_dim, name='z_mean', activation='linear')(x_inputs)
z_log_var = Dense(latent_dim, name='z_log_var', activation='relu')(x_inputs)

encoder = Model(encoder_input, [z_mean, z_log_var], name='encoder')

decoder_input_mean = Input(shape=(latent_dim,), name="decoder_input_mean")
decoder_input_log_var = Input(shape=(latent_dim,), name="decoder_input_log_var")
latents = Concatenate()([decoder_input_mean, decoder_input_log_var])
x_decoder = Dense(256, activation='relu')(latents)
x_decoder = Dense(4 * 4 * 32, activation='relu')(x_decoder)
x_decoder = Reshape((4, 4, 32))(x_decoder)

x_decoder = Conv2DTranspose(filters=64, kernel_size=4, activation='relu', strides=2, padding='same')(x_decoder)
x_decoder = Conv2DTranspose(filters=32, kernel_size=4, activation='relu', strides=2, padding='same')(x_decoder)
x_decoder = Conv2DTranspose(filters=32, kernel_size=4, activation='relu', strides=2, padding='same')(x_decoder)

x_decoder = Conv2DTranspose(filters=6, kernel_size=1, strides=1, activation='linear', padding='same')(x_decoder)

decoder = Model([decoder_input_mean, decoder_input_log_var], x_decoder)

inputs = encoder_input

# instantiate darla_vae model
encoder_outputs = encoder(encoder_input) # z_mean, z_log_var
reconstruction = decoder([encoder_outputs[0], encoder_outputs[1]])
outputs = [reconstruction, z_mean, z_log_var]
darla_vae = Model(inputs, outputs, name='darla_vae')

# Adding loss function
# Reconstruction loss
sampled_reconstructions = sampling([outputs[0][:, :, :, :3], outputs[0][:, :, :, 3:]])

reconstruction_loss = K.mean((255. * sampled_reconstructions - 255. * encoder_input)**2)

# KL loss
beta = 175.
kl_loss = 1 + z_mean - K.square(z_log_var) - K.exp(z_mean)
kl_loss = K.mean(kl_loss, axis=-1)
kl_loss *= -0.5

# Add the loss
vae_loss = K.mean(reconstruction_loss) + K.mean(beta * kl_loss)
darla_vae.add_loss(vae_loss)

# Compile the temporal vae
learning_rate = 1e-4
adam = Adam(lr=learning_rate)
darla_vae.compile(optimizer=adam)

if __name__ == '__main__':
    img_generator = DataSequence()
    # darla_vae.load_weights("sanity_check_darla2.h5")
    if True:
        history = darla_vae.fit_generator(img_generator, epochs=epochs, workers=9)
        darla_vae.save_weights("sanity_check_darla2.h5")
        darla_vae.save("sanity_check_darla2.h5")
    json_string = darla_vae.to_json()
    with open("darla_vae_arch.json", "w") as json_file:
        json_file.write(json_string)


    if True: # Test the temporal autoencoder
# Load Data
        data = np.load(glob.glob("training_data_small2/*.npz")[0])
        observations = data["images"]

        predictions = darla_vae.predict(observations)
        predicted_means = predictions[-2]
        predicted_log_vars = predictions[-1]

        if True:
            for j in trange(0, 32):
                step_size = (predicted_means[:, j].max() - predicted_means[:, j].min()) / 50.
                predicted_min = predicted_means[:, j].min()
                predicted_originals = predicted_means[:, j]
                for i in range(51):
                    # Change the RP
                    v = (i * step_size) + predicted_min
                    predicted_means[:, j] = v

                    # Predict, decode, denoise, and write to file
                    predicted_imgs = decoder.predict([predicted_means, predicted_log_vars])[:, :, :, :3]
                    predicted_img = np.clip(predicted_imgs[35], 0., 1.)
                    str_i = str(i)
                    if i < 10:
                        str_i = "0" + str_i
                    str_j = str(j)
                    if j < 10:
                        str_j = "0" + str_j
                    cv2.imwrite("sweep/viz" + str_j + "_" + str_i + ".png", cv2.resize(predicted_img * 255., (512, 512)))
                predicted_means[:, j] = predicted_originals

        if False:
            for i in range(0, len(predictions[0]), 1000):
                img = observations[i]
                cv2.imwrite("originals/original" + str(i) + ".png", cv2.resize(img * 255., (512, 512)))
                predicted = darla_vae.predict(np.expand_dims(img, axis=0))[0][0, :, :, :3]
                cv2.imwrite("reconstructions/recon" + str(i) + ".png", cv2.resize(predicted * 255., (512, 512)))


        sys.exit()
# Print loss values
        rps = predictions[1]
        print("RP LOSS VALUE", np.mean((rps[1:, :] - rps[:-1, :])**2))
        ss = predictions[2]
        pss = predictions[3]
        print("PREDICTION LOSS VALUE", np.mean((ss[1:, :] - pss[:-1, :])**2))

# Plot RP, S
        rps = np.transpose(rps)
        for r in rps:
            plt.plot(r, color='b', alpha=0.1)
        ss = np.transpose(ss)
        for s in ss:
            plt.plot(s, color='r', alpha=0.1)
        plt.savefig("plot.png")
