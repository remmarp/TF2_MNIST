# -*- coding: utf-8 -*-
# """
# auto_encoder/inference.py
# """

############
#   IMPORT #
############
# 1. Built-in modules
import os
import sys

sys.path.append(os.getcwd())

# 2. Third-party modules
import numpy as np
import tensorflow as tf

# 3. Own modules
from data_loader import MNISTLoader
from visualize import save_decode_image_array
from auto_encoder.parameter import Parameter
from auto_encoder.networks import Encoder, Decoder

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


################
#   DEFINITION #
################
def inference(denoise=True):
    param = Parameter()

    # 1. Build models
    encoder = Encoder(param).model()
    decoder = Decoder(param).model()

    encoder.summary(line_length=param.model_display_len)
    decoder.summary(line_length=param.model_display_len)

    # 2. Load data
    data_loader = MNISTLoader()
    train_set = data_loader.train.batch(batch_size=param.batch_size, drop_remainder=True)
    test_set = data_loader.test.batch(batch_size=param.batch_size, drop_remainder=True)

    # 3. Etc.
    graph_path = os.path.join(param.cur_dir, 'graph')
    if not os.path.isdir(graph_path):
        os.mkdir(graph_path)

    model_path = os.path.join(param.cur_dir, 'model')
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    # 4. Load model
    if denoise is True:
        _enc = 'mse_denoise_ae_encoder'
        _dec = 'mse_denoise_ae_decoder'
        graph = 'denoising_ae'
    else:
        _enc = 'mse_ae_encoder'
        _dec = 'mse_ae_decoder'
        graph = 'ae'

    encoder.load_weights(os.path.join(model_path, _enc))
    decoder.load_weights(os.path.join(model_path, _dec))

    # 5. Define loss
    mse = tf.keras.losses.MeanSquaredError()

    # 6. Inference
    train_mse = []
    for x_train, _ in train_set:
        z = encoder(x_train, training=False)
        x_bar = decoder(z, training=False)

        loss_ae = tf.reduce_mean(mse(x_train, x_bar))

        train_mse.append(loss_ae.numpy())

    num_test = 0
    valid_mse = []
    test_mse = []
    for x_test, _ in test_set:
        z = encoder(x_test, training=False)
        x_bar = decoder(z, training=False)

        loss_ae = tf.reduce_mean(mse(x_test, x_bar))

        if num_test <= param.valid_step:
            valid_mse.append(loss_ae.numpy())
        else:
            test_mse.append(loss_ae.numpy())
        num_test += 1

    # 7. Report
    print("[Loss] Train: {:.06f}\t Validation: {:.06f}\t Test: {:.06f}".format(np.mean(train_mse), np.mean(valid_mse),
                                                                               np.mean(test_mse)))

    # 8. Draw some samples
    if denoise is True:
        save_decode_image_array(x_test.numpy(), path=os.path.join(graph_path, '{}_original.png'.format(graph)))
        noise = tf.random.normal(shape=(param.batch_size,) + param.input_dim, mean=0.0,
                                 stddev=param.white_noise_std, dtype=tf.float32)
        x_test = x_test + noise

        z = encoder(x_test, training=False)
        x_bar = decoder(z, training=False)

        save_decode_image_array(x_test.numpy(), path=os.path.join(graph_path, '{}_add_noise.png'.format(graph)))
        save_decode_image_array(x_bar.numpy(), path=os.path.join(graph_path, '{}_decoded.png'.format(graph)))
    else:
        save_decode_image_array(x_test.numpy(), path=os.path.join(graph_path, '{}_original.png'.format(graph)))
        save_decode_image_array(x_bar.numpy(), path=os.path.join(graph_path, '{}_decoded.png'.format(graph)))


if __name__ == '__main__':
    inference(denoise=False)
    inference(denoise=True)
