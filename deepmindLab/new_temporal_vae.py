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

epochs = 5

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

class DataSequence(Sequence):
    def __init__(self):
        self.filenames = glob.glob("vae_training_data/*")[0:100]
        self.image_size = 84
        self.curr_episode = 0
        self.i = 0

    def on_epoch_end(self):
        pass

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        data = np.load(self.filenames[self.i])
        observations = data["observations"]
        actions = np.expand_dims(data["actions"], axis=-1)
        batch_x = [observations, actions]
        # batch_y = np.copy(data["observations"])
        self.i = (self.i + 1) // len(self.filenames)

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
    epsilon = K.random_normal(shape=(batch, 84, 84, 3))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# network parameters
latent_dim = 32
rp_dim = int(np.round(latent_dim * 0.5))
s_dim = latent_dim - rp_dim
input_shape = (84, 84, 3)

# build encoder network
encoder_input = Input(shape=input_shape, name='encoder_input')
x_inputs = Conv2D(filters=32, kernel_size=4, activation='relu', strides=2, padding='same')(encoder_input)
x_inputs = Conv2D(filters=32, kernel_size=4, activation='relu', strides=2, padding='same')(x_inputs)
x_inputs = Conv2D(filters=64, kernel_size=4, activation='relu', strides=2, padding='same')(x_inputs)
x_inputs = Conv2D(filters=64, kernel_size=4, activation='relu', strides=2, padding='same')(x_inputs)

x_inputs = Flatten()(x_inputs)
x_inputs = Dense(256, activation='relu')(x_inputs)

z_mean = Dense(latent_dim, name='z_mean', activation='linear')(x_inputs)
z_log_var = Dense(latent_dim, name='z_log_var', activation='relu')(x_inputs)

encoder = Model(encoder_input, [z_mean, z_log_var])

decoder_input_mean = Input(shape=(latent_dim,), name="decoder_input_mean")
decoder_input_log_var = Input(shape=(latent_dim,), name="decoder_input_log_var")
latents = Concatenate()([decoder_input_mean, decoder_input_log_var])
x_decoder = Dense(256, activation='relu')(latents)
x_decoder = Dense(6 * 6 * 64, activation='relu')(x_decoder)
x_decoder = Reshape((6, 6, 64))(x_decoder)

x_decoder = Conv2DTranspose(filters=64, kernel_size=4, activation='relu', strides=2, padding='same')(x_decoder)
x_decoder = Conv2DTranspose(filters=64, kernel_size=4, activation='relu', strides=2, padding='same')(x_decoder)
x_decoder = Conv2DTranspose(filters=32, kernel_size=4, activation='relu', strides=2, padding='same')(x_decoder)
x_decoder = Conv2DTranspose(filters=32, kernel_size=4, activation='relu', strides=2, padding='same')(x_decoder)

x_decoder = Conv2DTranspose(filters=6, kernel_size=1, strides=1, activation='linear', padding='same')(x_decoder)
x_decoder = Lambda(lambda x: x[:, :84, :84, :])(x_decoder)

decoder = Model([decoder_input_mean, decoder_input_log_var], x_decoder)

# instantiate prediction network
prediction_input_mean = Input(shape=(latent_dim,), name="prediction_input_mean")
prediction_input_log_var = Input(shape=(latent_dim,), name="prediction_input_log_var")
a_input = Input(shape=(1,), name='action_inputs')

z_mean_rp = Lambda(lambda x: x[:, :rp_dim])(prediction_input_mean)
z_log_var_rp = Lambda(lambda x: x[:, :rp_dim])(prediction_input_log_var)
render_parameters = Concatenate()([z_mean_rp, z_log_var_rp])

z_mean_s = Lambda(lambda x: x[:, rp_dim:])(prediction_input_mean)
z_log_var_s = Lambda(lambda x: x[:, rp_dim:])(prediction_input_log_var)
state_parameters = Concatenate()([z_mean_s, z_log_var_s])

prediction_inputs = Concatenate()([state_parameters, a_input])
predicted_state_parameters = Dense(256, activation='relu')(prediction_inputs)
predicted_state_parameters = Dense(256, activation='relu')(predicted_state_parameters)
predicted_state_parameters = Dense(latent_dim, activation='relu')(predicted_state_parameters)

predictor = Model([prediction_input_mean, prediction_input_log_var, a_input], [predicted_state_parameters, render_parameters, state_parameters])

a_top_input = Input(shape=(1,), name="top_action_inputs")
inputs = [encoder_input, a_top_input]

# instantiate temporal_vae model
encoder_outputs = encoder(encoder_input) # z_mean, z_log_var
predictor_outputs = predictor([encoder_outputs[0], encoder_outputs[1], a_top_input])
reconstruction = decoder([encoder_outputs[0], encoder_outputs[1]])
outputs = [reconstruction, predictor_outputs[1], predictor_outputs[2], predictor_outputs[0], z_mean, z_log_var]
temporal_vae = Model(inputs, outputs, name='temporal_vae')

# Adding loss function
# Reconstruction loss
denoising = load_model('full_denoising_autoencoder.h5')
denoising.compile(loss='mse', optimizer='adam')

denoising_encoder = Model(denoising.inputs, [denoising.layers[-10].output])
for layer in denoising_encoder.layers:
    layer.trainable = False

sampled_reconstructions = sampling([outputs[0][:, :, :, :3], outputs[0][:, :, :, 3:]])

reconstruction_loss = K.mean((denoising_encoder(sampled_reconstructions) - denoising_encoder(encoder_input))**2)

# KL loss
beta = 1.
kl_loss = 1 + z_mean - K.square(z_log_var) - K.exp(z_mean)
kl_loss = K.mean(kl_loss, axis=-1)
kl_loss *= -0.5

# RP loss
# rp_loss = K.mean((outputs[1][:, :] - outputs[1][:-1, :])**2)
rp_loss = K.mean(K.var(outputs[1], axis=0))

# Prediction loss
prediction_loss = K.mean((outputs[2][1:, :] - outputs[3][:-1, :])**2)

# Add the loss
vae_loss = K.mean(reconstruction_loss) + K.mean(beta * kl_loss) + K.mean(rp_loss) + 0. * K.mean(prediction_loss)
temporal_vae.add_loss(vae_loss)

# Compile the temporal vae
learning_rate = 1e-4
adam = Adam(lr=learning_rate)
temporal_vae.compile(optimizer=adam)

if __name__ == '__main__':
    img_generator = DataSequence()
    # temporal_vae.load_weights("temporal_vae.h5")
    if True:
        history = temporal_vae.fit_generator(img_generator, epochs=epochs, workers=9)
        temporal_vae.save_weights("temporal_vae.h5")
        temporal_vae.save("full_temporal_vae.h5")


    if True: # Test the temporal autoencoder
# Load Data
        data = np.load(glob.glob("vae_training_data/*")[0])
        observations = data["observations"]
        actions = data["actions"]

        predictions = temporal_vae.predict([observations, actions])
        predicted_means = predictions[-2]
        predicted_log_vars = predictions[-1]

        if False:
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
                    denoised_predicted = denoising.predict(predicted_imgs)
                    denoised_imgs = np.clip(denoised_predicted[35], 0., 1.)
                    str_i = str(i)
                    if i < 10:
                        str_i = "0" + str_i
                    str_j = str(j)
                    if j < 10:
                        str_j = "0" + str_j
                    cv2.imwrite("sweep/denoised_temporal" + str_j + "_" + str_i + ".png", cv2.resize(denoised_imgs * 255., (512, 512)))
                predicted_means[:, j] = predicted_originals

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
