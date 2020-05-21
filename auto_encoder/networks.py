# -*- coding: utf-8 -*-
# """
# auto_encoder/networks.py
# """

############
#   IMPORT #
############
# 1. Built-in modules

# 2. Third-party modules
import tensorflow as tf
import tensorflow.keras.layers as layers

# import torch
# import torchvision
# from torch import nn
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.datasets import MNIST
# from torchvision.utils import save_image

# 3. Own modules


###########
#   CLASS #
###########
class Encoder(object):
    def __init__(self, param):
        self.param = param

    def model(self):
        model = tf.keras.Sequential(name='Encoder')

        # Layer 1
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False,
                                input_shape=self.param.input_dim, name='l1_conv'))
        model.add(layers.BatchNormalization(name='l1_bn'))
        model.add(layers.LeakyReLU(name='l1_leaky'))
        assert model.output_shape == (None, 14, 14, 64)

        # Layer 2
        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False, name='l2_conv'))
        model.add(layers.BatchNormalization(name='l2_bn'))
        model.add(layers.LeakyReLU(name='l2_leaky'))
        assert model.output_shape == (None, 7, 7, 128)

        # Layer 3
        model.add(layers.Conv2D(256, (5, 5), strides=(1, 1), padding='same', use_bias=False, name='l3_conv'))
        model.add(layers.BatchNormalization(name='l3_bn'))
        model.add(layers.LeakyReLU(name='l3_leaky'))
        model.add(layers.Flatten(name='l3_flat'))
        assert model.output_shape == (None, 7 * 7 * 256)

        # Layer 4
        model.add(layers.Dense(self.param.latent_dim, use_bias=False, activation='tanh', name='l4_dense'))
        assert model.output_shape == (None, self.param.latent_dim)

        model.summary(line_length=self.param.model_display_len)

        return model


class Decoder(object):
    def __init__(self, param):
        self.param = param

    def model(self):
        model = tf.keras.Sequential(name='Decoder')

        # Layer 1
        model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(self.param.latent_dim,), name='l1_dense'))
        model.add(layers.BatchNormalization(name='l1_bn'))
        model.add(layers.LeakyReLU(name='l1_leaky'))
        model.add(layers.Reshape((7, 7, 256), name='l1_reshape'))
        assert model.output_shape == (None, 7, 7, 256)

        # Layer 2
        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False, name='l2_deconv'))
        model.add(layers.BatchNormalization(name='l2_bn'))
        model.add(layers.LeakyReLU(name='l2_leaky'))
        assert model.output_shape == (None, 7, 7, 128)

        # Layer 3
        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, name='l3_deconv'))
        model.add(layers.BatchNormalization(name='l3_bn'))
        model.add(layers.LeakyReLU(name='l3_leaky'))
        assert model.output_shape == (None, 14, 14, 64)

        # Layer 4
        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh',
                                         name='l4_deconv'))
        assert model.output_shape == (None, 28, 28, 1)

        model.summary(line_length=self.param.model_display_len)

        return model


if __name__ == '__main__':
    from auto_encoder.parameter import Parameter

    param = Parameter()

    encoder = Encoder(param).model()
    decoder = Decoder(param).model()
