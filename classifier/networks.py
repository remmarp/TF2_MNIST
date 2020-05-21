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
class Classifier(object):
    def __init__(self, param):
        self.param = param

    def model(self):
        model = tf.keras.Sequential(name='Classifier')

        # Layer 1
        model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', use_bias=False,
                                input_shape=self.param.input_dim, name='l1_conv'))
        model.add(layers.LeakyReLU(name='l1_leaky'))
        model.add(layers.Dropout(0.1, name='l1_drop'))
        assert model.output_shape == (None, 14, 14, 64)

        # Layer 2
        model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', use_bias=False, name='l2_conv'))
        model.add(layers.LeakyReLU(name='l2_leaky'))
        model.add(layers.Dropout(0.1, name='l2_drop'))
        model.add(layers.Flatten(name='l2_flat'))
        assert model.output_shape == (None, 7 * 7 * 128)

        # Layer 3
        model.add(layers.Dense(self.param.num_class, name='l3_dense'))
        assert model.output_shape == (None, self.param.num_class)

        model.summary(line_length=self.param.model_display_len)

        return model


if __name__ == '__main__':
    from classifier.parameter import Parameter

    param = Parameter()

    encoder = Classifier(param).model()
