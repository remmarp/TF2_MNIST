# -*- coding: utf-8 -*-
# """
# adversarial_autoencoder/networks.py
#   Supervised AAE: Disentagling the label information form the hidden code by providing the one-hot vector
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
class Encoder(object):
    def __init__(self, param):
        self.param = param

    def model(self):
        model = tf.keras.Sequential(name='Encoder')

        # Layer 1
        model.add(layers.Flatten(input_shape=(28, 28, 1), name='l1_flatten'))
        model.add(layers.Dense(1024, activation=tf.keras.activations.relu, name='l1_dense'))
        model.add(layers.BatchNormalization(name='l1_bn'))
        assert model.output_shape == (None, 1024)

        # Layer 2
        model.add(layers.Dense(512, activation=tf.keras.activations.relu, name='l2_dense'))
        model.add(layers.BatchNormalization(name='l2_bn'))
        assert model.output_shape == (None, 512)

        # Layer 3
        model.add(layers.Dense(256, activation=tf.keras.activations.relu, name='l3_dense'))
        model.add(layers.BatchNormalization(name='l3_bn'))
        assert model.output_shape == (None, 256)

        # Layer 4
        model.add(layers.Dense(self.param.latent_dim, activation='tanh', name='l4_dense'))

        assert model.output_shape == (None, self.param.latent_dim)

        # Layer 1
        # model = tf.keras.Sequential()
        # model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False,
        #                         input_shape=self.param.input_dim, name='l1_conv'))
        # model.add(layers.BatchNormalization(name='l1_bn'))
        # model.add(layers.LeakyReLU(name='l1_leaky'))
        # assert model.output_shape == (None, 14, 14, 64)
        #
        # # Layer 2
        # model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False, name='l2_conv'))
        # model.add(layers.BatchNormalization(name='l2_bn'))
        # model.add(layers.LeakyReLU(name='l2_leaky'))
        # assert model.output_shape == (None, 7, 7, 128)
        #
        # # Layer 3
        # model.add(layers.Conv2D(256, (5, 5), strides=(1, 1), padding='same', use_bias=False, name='l3_conv'))
        # model.add(layers.BatchNormalization(name='l3_bn'))
        # model.add(layers.LeakyReLU(name='l3_leaky'))
        # model.add(layers.Flatten(name='l3_flat'))
        # assert model.output_shape == (None, 7 * 7 * 256)
        #
        # # Layer 4
        # model.add(layers.Dense(self.param.latent_dim, activation='tanh', name='l4_dense'))

        model.summary(line_length=self.param.model_display_len)

        return model


class Decoder(object):
    def __init__(self, param):
        self.param = param

    def model(self):
        model = tf.keras.Sequential(name='Decoder')

        # Layer 1
        model.add(layers.Dense(256, activation=tf.keras.activations.relu, input_shape=(self.param.latent_dim + self.param.num_class,),
                               name='l1_dense'))
        model.add(layers.BatchNormalization(name='l1_bn'))
        assert model.output_shape == (None, 256)

        # Layer 2
        model.add(layers.Dense(512, activation=tf.keras.activations.relu, name='l2_dense'))
        model.add(layers.BatchNormalization(name='l2_bn'))
        assert model.output_shape == (None, 512)

        # Layer 3
        model.add(layers.Dense(1024, activation=tf.keras.activations.relu,
                               input_shape=(self.param.latent_dim + self.param.num_class,),
                               name='l3_dense'))
        model.add(layers.BatchNormalization(name='l3_bn'))
        assert model.output_shape == (None, 1024)

        # Layer 3
        model.add(layers.Dense(28*28, activation=tf.keras.activations.tanh, name='l4_dense'))
        model.add(layers.Reshape((28, 28, 1), name='l4_reshape'))
        assert model.output_shape == (None, 28, 28, 1)
        #
        # Layer 1
        # model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(self.param.latent_dim + self.param.num_class,),
        #                        name='l1_dense'))
        # model.add(layers.BatchNormalization(name='l1_bn'))
        # model.add(layers.LeakyReLU(name='l1_leaky'))
        # model.add(layers.Reshape((7, 7, 256), name='l1_reshape'))
        # assert model.output_shape == (None, 7, 7, 256)
        #
        # # Layer 2
        # model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False, name='l2_deconv'))
        # model.add(layers.BatchNormalization(name='l2_bn'))
        # model.add(layers.LeakyReLU(name='l2_leaky'))
        # assert model.output_shape == (None, 7, 7, 128)
        #
        # # Layer 3
        # model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, name='l3_deconv'))
        # model.add(layers.BatchNormalization(name='l3_bn'))
        # model.add(layers.LeakyReLU(name='l3_leaky'))
        # assert model.output_shape == (None, 14, 14, 64)
        #
        # # Layer 4
        # model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh',
        #                                  name='l4_deconv'))
        # assert model.output_shape == (None, 28, 28, 1)

        model.summary(line_length=self.param.model_display_len)

        return model


class Discriminator(object):
    def __init__(self, param):
        self.param = param

    def model(self):
        model = tf.keras.Sequential(name='Discriminator')

        # Layer 1
        model.add(layers.Dense(128, input_shape=(self.param.latent_dim + self.param.num_class, ), name='l1_dense'))
        model.add(layers.LeakyReLU(name='l1_leaky'))
        model.add(layers.Dropout(0.1, name='l1_drop'))
        assert model.output_shape == (None, 128)

        # Layer 2
        model.add(layers.Dense(32, name='l2_dense'))
        model.add(layers.LeakyReLU(name='l2_leaky'))
        model.add(layers.Dropout(0.1, name='l2_drop'))
        assert model.output_shape == (None, 32)

        # Layer 3
        model.add(layers.Dense(1, use_bias=True, name='l3_dense'))
        assert model.output_shape == (None, 1)

        model.summary(line_length=self.param.model_display_len)

        return model


if __name__ == '__main__':
    from adversarial_autoencoder.parameter import Parameter

    param = Parameter()

    encoder = Encoder(param).model()
    decoder = Decoder(param).model()
    discriminator = Discriminator(param).model()
