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
        model.add(layers.Dense(256, use_bias=True, input_shape=(self.param.num_class + self.param.latent_dim, ),
                               name='l1_dense'))
        model.add(layers.BatchNormalization(name='l1_bn'))
        model.add(layers.LeakyReLU(name='l1_leaky'))
        assert model.output_shape == (None, 256)

        # Layer 2
        model.add(layers.Dense(512, use_bias=True, name='l2_dense'))
        model.add(layers.BatchNormalization(name='l2_bn'))
        model.add(layers.LeakyReLU(name='l2_leaky'))
        assert model.output_shape == (None, 512)

        # Layer 3
        model.add(layers.Dense(1024, use_bias=True, name='l3_dense'))
        model.add(layers.BatchNormalization(name='l3_bn'))
        model.add(layers.LeakyReLU(name='l3_leaky'))
        assert model.output_shape == (None, 1024)

        # Layer 4
        model.add(layers.Dense(2048, use_bias=True, name='l4_dense'))
        model.add(layers.BatchNormalization(name='l4_bn'))
        model.add(layers.LeakyReLU(name='l4_leaky'))
        assert model.output_shape == (None, 2048)

        # Layer 5
        model.add(layers.Dense(784, use_bias=True, activation='tanh', name='l5_dense'))
        model.add(layers.Reshape((28, 28, 1), name='l5_reshape'))
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
        model.add(layers.Flatten(input_shape=(28, 28, 1), name='l1_flatten'))
        model.add(layers.Dense(2048, use_bias=True, name='l1_dense'))
        model.add(layers.LeakyReLU(name='l1_leaky'))
        assert model.output_shape == (None, 2048)

        # Layer 2
        model.add(layers.Dense(1024, use_bias=True, name='l2_dense'))
        model.add(layers.LeakyReLU(name='l2_leaky'))
        assert model.output_shape == (None, 1024)

        # Layer 3
        model.add(layers.Dense(512, use_bias=True, name='l3_dense'))
        model.add(layers.LeakyReLU(name='l3_leaky'))
        assert model.output_shape == (None, 512)

        # Layer 4
        model.add(layers.Dense(256, use_bias=True, name='l4_dense'))
        model.add(layers.LeakyReLU(name='l4_leaky'))
        assert model.output_shape == (None, 256)

        # Layer 5
        model.add(layers.Dense(1, use_bias=True, activation=None, name='l5_dense'))
        assert model.output_shape == (None, 1)

        out = model.get_layer(name='l4_leaky').output
        dis = model.get_layer(name='l5_dense').output

        _model = tf.keras.Model(model.input, outputs=[out, dis], name='Discriminator')

        _model.summary(line_length=self.param.model_display_len)

        return _model


class Classifier(tf.keras.Model):
    def __init__(self, param):
        super(Classifier, self).__init__()

        self.param = param

    def model(self):
        model = tf.keras.Sequential(name='Classifier')

        # Layer 1
        model.add(layers.Dense(1024, use_bias=True, input_shape=(256, ), name='l1_dense'))
        model.add(layers.LeakyReLU(name='l1_leaky'))
        model.add(layers.Dropout(rate=0.1, name='l1_dropout'))
        assert model.output_shape == (None, 1024)

        # Layer 2
        model.add(layers.Dense(512, use_bias=True, name='l2_dense'))
        model.add(layers.LeakyReLU(name='l2_leaky'))
        model.add(layers.Dropout(rate=0.1, name='l2_dropout'))
        assert model.output_shape == (None, 512)

        # Layer 3
        model.add(layers.Dense(self.param.num_class, use_bias=True, name='l3_dense'))
        assert model.output_shape == (None, self.param.num_class)

        model.summary(line_length=self.param.model_display_len)

        return model


if __name__ == '__main__':
    from acgans.parameter import Parameter

    param = Parameter()

    generator = Generator(param).model()
    discriminator = Discriminator(param).model()
    classifier = Classifier(param).model()
