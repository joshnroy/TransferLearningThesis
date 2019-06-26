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

from variational_autoencoder_deconv import vae, inputs, outputs


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


imgs = np.asarray([cv2.imread(x) for x in glob.glob("training_observations_jaco/*.png")])
x_train = imgs[150:]
x_test = imgs[:150]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = np.array([])
y_test = np.array([])
image_size = 64
input_shape = (image_size, image_size, 3)

# network parameters
kernel_size = 3
latent_dim = 64
rp_dim = 32
s_dim = latent_dim - rp_dim
num_conv = 3

rp_l2_loss_weight = 1000.
s_l2_loss_weight = 1.
s_l0_loss_weight = 1.

batch_size = 128
epochs = 1150

# # build encoder model
# inputs = Input(shape=input_shape, name='encoder_input')
# x_inputs = Conv2D(filters=32, kernel_size=4, activation='relu', strides=2,
#                   padding='same')(inputs)
# x_inputs = Conv2D(filters=32, kernel_size=4, activation='relu', strides=2,
#                   padding='same')(x_inputs)
# x_inputs = Conv2D(filters=64, kernel_size=4, activation='relu', strides=2,
#                   padding='same')(x_inputs)
# x_inputs = Conv2D(filters=64, kernel_size=4, activation='relu', strides=2,
#                   padding='same')(x_inputs)

# x_inputs = Flatten()(x_inputs)
# x_inputs = Dense(256, activation='relu')(x_inputs)
# z_mean = Dense(latent_dim, name='z_mean', activation='linear')(x_inputs)
# z_log_var = Dense(latent_dim, name='z_log_var', activation='linear')(x_inputs)

# z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])





# # instantiate encoder model
# encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
# encoder.summary()






# # build decoder model
# latent_inputs = Input(shape=(latent_dim,))
# x_decoder = Dense(256, activation='relu')(latent_inputs)
# x_decoder = Dense(4 * 4 * 64, activation='relu')(x_decoder)
# x_decoder = Reshape((4, 4, 64))(x_decoder)

# x_decoder = Conv2DTranspose(filters=64, kernel_size=4, activation='relu',
#                             strides=2, padding='same')(x_decoder)
# x_decoder = Conv2DTranspose(filters=64, kernel_size=4, activation='relu',
#                             strides=2, padding='same')(x_decoder)
# x_decoder = Conv2DTranspose(filters=32, kernel_size=4, activation='relu',
#                             strides=2, padding='same')(x_decoder)
# x_decoder = Conv2DTranspose(filters=32, kernel_size=4, activation='relu',
#                             strides=2, padding='same')(x_decoder)

# x_decoder = Conv2DTranspose(filters=3, kernel_size=1, strides=1,
#                             activation='sigmoid', padding='same')(x_decoder)





# # instantiate decoder model
# decoder = Model(latent_inputs, x_decoder, name='decoder')
# # for layer in decoder.layers:
# #     layer.name += "_decoder"
# decoder.summary()




# # instantiate VAE model
# encoder_outputs = encoder(inputs)
# outputs = [encoder_outputs[0], encoder_outputs[1], decoder(encoder_outputs[2])]
# vae = Model(inputs, outputs, name='vae')

if __name__ == '__main__':
    global vae
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m", "--mse", help=help_, action='store_true')
    args = parser.parse_args()
    # models = (encoder, decoder)
    data = (x_test, y_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs[2]))
    else:
        reconstruction_loss = binary_crossentropy(K.flatten(inputs),
                                                  K.flatten(outputs))

    reconstruction_loss *= image_size * image_size
    beta = 175.
    kl_loss = 1 + outputs[0] - K.square(outputs[1]) - K.exp(outputs[0])
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5


    z = outputs[2]
    rp_l2_loss = rp_l2_loss_weight * K.mean(K.square(z[1:, :rp_dim] - z[:-1, :rp_dim]))
    s_l2_loss = -1 * s_l2_loss_weight * K.mean(K.square(z[1:, s_dim:] - z[:-1, s_dim:]))
    s_difference = K.abs(z[1:, s_dim:] - z[:-1, s_dim:]) + 1e-5
    s_difference /= K.max([K.max(s_difference), 1e-5])
    normalized_s = s_difference
    s_l0_loss = s_l0_loss_weight * -1 * K.mean((normalized_s + 1e-5) *
                                               (K.log((normalized_s + 1e-5))))

    vae_loss = K.mean(reconstruction_loss + beta * kl_loss + rp_l2_loss +
                      s_l2_loss + s_l0_loss)
    vae.add_loss(vae_loss)

    learning_rate = 1e-4
    adam = Adam(lr=learning_rate)
    vae.compile(optimizer=adam)
    vae.summary()

    if args.weights:
        pass
    else:
        # train the autoencoder
        if True:
            print("starting to generate all images")
            all_images = [np.asarray([cv2.imread(x) for x in
                                      glob.glob("training_observations_jaco/obs" +
                                                str(i) + "_*.png")]) / 255. for i
                          in trange(999)]
            print("loaded all images")
            epoch_losses = []
            for epoch in range(epochs):
                losses = []
                for i in trange(999):
                    imgs_batch = all_images[i]
                    np.random.shuffle(imgs_batch)
                    loss = vae.train_on_batch(imgs_batch, y=None)
                    # history = vae.fit(imgs_batch, epochs=1, batch_size=batch_size, verbose=0)
                    # loss = history.history
                    # print(loss)
                    losses.append(loss)
                print("epoch", epoch, "\tmean loss", np.mean(losses))
                epoch_losses.append(np.mean(losses))
                if epoch % 10 == 0:
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
                        # np.random.shuffle(all_images[i])
                        imgs_batch = all_images[i]
                        outputs = encoder.predict_on_batch(imgs_batch)
                        predicted_z = outputs[2]
                        normalized_s = np.abs(predicted_z[:, s_dim:]) / (np.sum(np.abs(predicted_z[:, s_dim:])) + 1e-5)
                        print("\n")
                        print("rp L2 loss", np.mean(rp_l2_loss_weight * (predicted_z[1:, :rp_dim] - predicted_z[:-1, :rp_dim])**2),
                              "s L2 loss", np.mean(s_l2_loss_weight * (predicted_z[1:, s_dim:] - predicted_z[:-1, s_dim:])**2),
                              "s Entropy loss", np.mean(s_l0_loss_weight * -1 * (normalized_s + 1e-5) * np.log((normalized_s + 1e-5))))
                        print("RP Min", np.min(predicted_z[0, :rp_dim]),"RP Mean", np.mean(predicted_z[0, :rp_dim]), "RP Max", np.max(predicted_z[0, :rp_dim]))
                        print("S Min", np.min(predicted_z[0, s_dim:]), "S Mean", np.mean(predicted_z[0, s_dim:]), "S Max", np.max(predicted_z[0, s_dim:]))
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
