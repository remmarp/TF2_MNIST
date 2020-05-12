# -*- coding: utf-8 -*-
# """
# data_loader.py
# """

############
#   IMPORT #
############
# 1. Built-in modules

# 2. Third-party modules
import numpy as np
import tensorflow as tf

# 3. Own modules


###########
#   CLASS #
###########
class MNISTLoader(object):
    def __init__(self):
        # Load MNIST
        data_url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

        path = tf.keras.utils.get_file('mnist.npz', data_url)

        with np.load(path) as data:
            x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']

        self.num_train = len(x_train)
        self.num_test = len(x_test)

        _x_train = tf.convert_to_tensor(np.expand_dims(x_train / 255., axis=-1), dtype=tf.float32)
        _x_test = tf.convert_to_tensor(np.expand_dims(x_test / 255., axis=-1), dtype=tf.float32)
        _y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
        _y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

        x_train_set = tf.data.Dataset.from_tensor_slices(_x_train)
        y_train_set = tf.data.Dataset.from_tensor_slices(_y_train)

        x_test_set = tf.data.Dataset.from_tensor_slices(_x_test)
        y_test_set = tf.data.Dataset.from_tensor_slices(_y_test)

        self.train = tf.data.Dataset.zip((x_train_set, y_train_set))
        self.test = tf.data.Dataset.zip((x_test_set, y_test_set))


class NoveltyDetectionMNISTLoader(object):
    def __init__(self, cls=0):
        # Load MNIST
        data_url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'

        path = tf.keras.utils.get_file('mnist.npz', data_url)
        with np.load(path) as data:
            x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']

        train_data_tensor = np.expand_dims(x_train[y_train == cls] / 255., axis=-1)
        self.train = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(train_data_tensor, dtype=tf.float32))

        test_data_tensor = np.expand_dims(x_test / 255., axis=-1)
        test_data = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(test_data_tensor, dtype=tf.float32))

        test_label_tensor = (y_test == cls)
        test_label = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(test_label_tensor, dtype=tf.int32))

        self.test = tf.data.Dataset.zip((test_data, test_label))
