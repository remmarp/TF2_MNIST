# -*- coding: utf-8 -*-
# """
# generative_adversarial_networks/networks.py
# """

############
#   IMPORT #
############
# 1. Built-in modules

# 2. Third-party modules
import tensorflow as tf
import tensorflow.keras.layers as layers

# 3. Own modules


###########
#   CLASS #
###########
class Generator(object):
    def __init__(self, param):
        self.param = param

    def model(self):
        model = tf.keras.Sequential(name='Generator')

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


class Discriminator(tf.keras.Model):
    def __init__(self, param):
        super(Discriminator, self).__init__()

        self.param = param

    def model(self):
        model = tf.keras.Sequential(name='Discriminator')

        # Layer 1
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False,
                                input_shape=self.param.input_dim, name='l1_conv'))
        model.add(layers.LeakyReLU(name='l1_leaky'))
        model.add(layers.Dropout(0.3, name='l1_drop'))
        assert model.output_shape == (None, 14, 14, 64)

        # Layer 2
        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False, name='l2_deconv'))
        model.add(layers.LeakyReLU(name='l2_leaky'))
        model.add(layers.Dropout(0.3, name='l2_drop'))
        model.add(layers.Flatten(name='l2_flat'))
        assert model.output_shape == (None, 7 * 7 * 128)

        # Layer 3
        model.add(layers.Dense(1, use_bias=True, name='l3_dense'))
        assert model.output_shape == (None, 1)

        model.summary(line_length=self.param.model_display_len)

        return model


if __name__ == '__main__':
    from generative_adversarial_networks.parameter import Parameter

    param = Parameter()

    generator = Generator(param).model()
    discriminator = Discriminator(param).model()
