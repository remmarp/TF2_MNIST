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
class Classifier(tf.keras.Model):
    def __init__(self, param):
        super(Classifier, self).__init__()

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

        # [None, 7 * 7 * 128] -> [None, 10]
        self.l3_dense = layers.Dense(param.num_class, name='l3_dense')


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

