# -*- coding: utf-8 -*-
# """
# classifier/networks.py
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
class Generator(tf.keras.Model):
    def __init__(self, param):
        super(Generator, self).__init__()
        self.param = param

        # [None, 256] -> [None, 7, 7, 256]
        self.l1_dense = layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(param.latent_dim, ), name='l1_dense')
        self.l1_bn = layers.BatchNormalization(name='l1_bn')
        self.l1_leaky = layers.LeakyReLU(name='l1_leaky')
        self.l1_reshape = layers.Reshape((7, 7, 256))

        # [None, 7, 7, 256] -> [None, 7, 7, 128]
        self.l2_deconv = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False,
                                                name='l2_deconv')
        self.l2_bn = layers.BatchNormalization(name='l2_bn')
        self.l2_leaky = layers.LeakyReLU(name='l2_leaky')

        # [None, 7, 7, 128] -> [None, 14, 14, 64]
        self.l3_deconv = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False,
                                                name='l3_deconv')
        self.l3_bn = layers.BatchNormalization(name='l3_bn')
        self.l3_leaky = layers.LeakyReLU(name='l3_leaky')

        # [None, 14, 14, 64] -> [None, 28, 28, 1]
        self.l4_deconv = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False,
                                                activation=tf.keras.activations.tanh, name='l4_deconv')


    def call(self, inputs, training=False):
        l1 = self.l1_dense(inputs)
        if training is True:
            l1 = self.l1_bn(l1)
        l1 = self.l1_leaky(l1)
        l1 = self.l1_reshape(l1)

        l2 = self.l2_deconv(l1)
        if training is True:
            l2 = self.l2_bn(l2)
        l2 = self.l2_leaky(l2)

        l3 = self.l3_deconv(l2)
        if training is True:
            l3 = self.l3_bn(l3)
        l3 = self.l3_leaky(l3)

        l4 = self.l4_deconv(l3)

        return l4

    def model(self):
        z = tf.keras.Input(shape=self.param.latent_dim, dtype=tf.float32, name='z')
        return tf.keras.Model(inputs=z, outputs=self.call(z))


class Discriminator(tf.keras.Model):
    def __init__(self, param):
        super(Discriminator, self).__init__()

        self.param = param

        # [None, 28, 28, 1] -> [None, 14, 14, 64]
        self.l1_conv = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False,
                                     input_shape=param.input_dim, name='l1_conv')
        self.l1_leaky = layers.LeakyReLU(name='l1_leaky')
        self.l1_drop = layers.Dropout(0.3, name='l1_drop')

        # [None, 14, 14, 64] -> [None, 7 * 7 * 128]
        self.l2_conv = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False, name='l2_conv')
        self.l2_leaky = layers.LeakyReLU(name='l2_leaky')
        self.l2_drop = layers.Dropout(0.3, name='l2_drop')
        self.l2_flat = layers.Flatten(name='l2_flat')

        # [None, 7 * 7 * 128] -> [None, 1]
        self.l3_dense = layers.Dense(1, name='l3_dense')

    def call(self, inputs, training=False):
        l1 = self.l1_conv(inputs)
        l1 = self.l1_leaky(l1)
        if training is True:
            l1 = self.l1_drop(l1)

        l2 = self.l2_conv(l1)
        l2 = self.l2_leaky(l2)
        if training is True:
            l2 = self.l2_drop(l2)
        l2 = self.l2_flat(l2)

        l3 = self.l3_dense(l2)

        return l3

    def model(self):
        x = tf.keras.Input(shape=self.param.input_dim, dtype=tf.float32, name='x')
        return tf.keras.Model(inputs=x, outputs=self.call(x))
