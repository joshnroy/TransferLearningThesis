'''Example of VAE on MNIST dataset using CNN

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.

# Reference

[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
import cv2
import sys
from tqdm import trange


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
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


# MNIST dataset
if False:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image_size = x_train.shape[1]
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    input_shape = (image_size, image_size, 1)
else:
    # deepmind Dataset
    imgs = np.asarray([cv2.imread(x) for x in glob.glob("training_observations_jaco/*.png")])
    x_train = imgs[:-100]
    x_test = imgs[-100:]
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    y_train = np.asarray([])
    y_test = np.asarray([])
    image_size = x_train.shape[1]
    input_shape = (image_size, image_size, 3)

# network parameters
kernel_size = 3
latent_dim = 32
rp_dim = 16
s_dim = latent_dim - rp_dim
num_conv = 3

rp_l2_loss_weight = 100.
s_l2_loss_weight = 1.
s_l0_loss_weight = 10.

batch_size = 128
epochs = 670

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
z_mean = Dense(latent_dim, name='z_mean', activation='linear')(x_inputs)
z_log_var = Dense(latent_dim, name='z_log_var', activation='linear')(x_inputs)

z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])





# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()






# build decoder model
latent_inputs = Input(shape=(latent_dim,))
x_decoder = Dense(256, activation='relu')(latent_inputs)
x_decoder = Dense(4 * 4 * 64, activation='relu')(x_decoder)
x_decoder = Reshape((4, 4, 64))(x_decoder)

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





# instantiate decoder model
decoder = Model(latent_inputs, x_decoder, name='decoder')
decoder.summary()




# instantiate VAE model
encoder_outputs = encoder(inputs)
outputs = [encoder_outputs[0], encoder_outputs[1], decoder(encoder_outputs[2])]
vae = Model(inputs, outputs, name='vae')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m", "--mse", help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    data = (x_test, y_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs[2]))
    else:
        reconstruction_loss = binary_crossentropy(K.flatten(inputs),
                                                  K.flatten(outputs))

    reconstruction_loss *= image_size * image_size

    beta = 175.
    kl_loss = 1 + outputs[1] - K.square(outputs[0]) - K.exp(outputs[1])
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5


    z = outputs[2]
    rp_l2_loss = rp_l2_loss_weight * K.mean(K.square(z[1:, :rp_dim] - z[:-1, :rp_dim]))
    s_l2_loss = s_l2_loss_weight * K.mean(K.square(z[1:, s_dim:] - z[:-1, s_dim:]))
    s_difference = K.abs(z[1:, s_dim:] - z[:-1, s_dim:]) + 1e-5
    s_difference /= K.max([K.max(s_difference), 1e-5])
    # normalized_s = K.abs(z[:, s_dim:]) / (K.sum(K.abs(z[:, s_dim:])) + 1e-5)
    normalized_s = s_difference
    s_l0_loss = s_l0_loss_weight * -1 * K.mean((normalized_s + 1e-5) * (K.log((normalized_s + 1e-5))))

    vae_loss = K.mean(reconstruction_loss + beta * kl_loss + rp_l2_loss - s_l2_loss + s_l0_loss)
    vae.add_loss(vae_loss)

    learning_rate = 1e-4
    adam = Adam(lr=learning_rate)
    vae.compile(optimizer=adam)
    vae.summary()
    # plot_model(vae, to_file='factored_vae.png', show_shapes=True)

    if args.weights:
        pass
    else:
        # train the autoencoder
        print("starting to generate all images")
        all_images = [np.asarray([cv2.imread(x) for x in glob.glob("training_observations_jaco/obs" + str(i) + "_*.png")]) / 255. for i in trange(999)]
        print("loaded all images")
        if True:
            epoch_losses = []
            for epoch in range(epochs):
                losses = []
                for i in trange(999):
                    imgs_batch = all_images[i]
                    loss = vae.train_on_batch(imgs_batch, y=None)
                    losses.append(loss)
                print("epoch", epoch, "\tmean loss", np.mean(losses))
                epoch_losses.append(np.mean(losses))
                if epoch % 50 == 0:
                    print("testing")
                    vae.save_weights('jaco_myvae_weights.h5')
                    predicted_imgs = vae.predict(x_test, batch_size=batch_size)
                    x_test_scaled = 255. * x_test
                    predicted_imgs_scaled = 255. * predicted_imgs[2]
                    for i in range(len(predicted_imgs_scaled)):
                        cv2.imwrite("images_updated/original_" + str(epoch) + "_" + str(i) + ".png", x_test_scaled[i])
                        cv2.imwrite("images_updated/reconstructed_" + str(epoch) + "_" + str(i) + ".png", predicted_imgs_scaled[i])
                    encoder = Model(vae.inputs, vae.layers[-2].outputs)
                    encoder.compile(optimizer=adam, loss='mse')
                    for i in range(2):
                        imgs_batch = all_images[i]
                        outputs = encoder.predict_on_batch(imgs_batch)
                        predicted_z = outputs[2]
                        normalized_s = np.abs(predicted_z[:, s_dim:]) / (np.sum(np.abs(predicted_z[:, s_dim:])) + 1e-5)
                        print("\n")
                        print("rp L2 loss", np.mean(rp_l2_loss_weight * (predicted_z[1:, :rp_dim] - predicted_z[:-1, :rp_dim])**2),
                              "s L2 loss", np.mean(s_l2_loss_weight * (predicted_z[1:, s_dim:] - predicted_z[:-1, s_dim:])**2),
                              "s Entropy loss", np.mean(s_l0_loss_weight * -1 * (normalized_s + 1e-5) * np.log((normalized_s + 1e-5))))
                        print("RP Min", np.min(predicted_z[0, :rp_dim]),"RP Mean", np.mean(predicted_z[0, :rp_dim]), "RP Max", np.max(predicted_z[0, :rp_dim]))
                        print("S Min", np.min(predicted_z[0, s_dim:]), "S Max", np.max(predicted_z[0, s_dim:]), "S Mean", np.mean(predicted_z[0, s_dim:]))
                        print("\n")

            np.savez_compressed("jaco_vae_history", epoch_losses=np.asarray(epoch_losses))
        else:
            vae.fit(x_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(x_test, None))
        vae.save_weights('jaco_myvae_weights.h5')

# Test the autoencoder
    # if True:
    #     predicted_imgs = vae.predict(x_test, batch_size=batch_size)
    #     x_test *= 255.
    #     predicted_imgs *= 255.
    #     for i in range(len(predicted_imgs)):
    #         cv2.imwrite("images/original" + str(i) + ".png", x_test[i])
    #         cv2.imwrite("images/reconstructed" + str(i) + ".png", predicted_imgs[i])
    # else:
    #     test_losses = 0.
    #     for i in range(99):
    #         imgs = np.asarray([cv2.imread(x) for x in glob.glob("training_observations_jaco/obs" + str(i) + "_*.jpg")])
    #         imgs_batch = imgs / 255.
    #         loss = vae.predict_on_batch(imgs_batch)
    #         # print("epoch", epoch, "episode", i, "loss", loss, "batch size", imgs_batch.shape[0])
    #         test_losses.append(loss)

    # # plot_results(models, data, batch_size=batch_size, model_name="factored_vae")
