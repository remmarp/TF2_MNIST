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
        graph = 'denoising_ae'
        enc_name = 'encoder_denoise'
        dec_name = 'decoder_denoise'
    else:
        graph = 'ae'
        enc_name = 'encoder'
        dec_name = 'decoder'

    encoder.load_weights(os.path.join(model_path, enc_name))
    decoder.load_weights(os.path.join(model_path, dec_name))

    # 5. Define loss

    # 6. Define test step ##############################################################################################
    def testing_step(_x):
        if denoise is True:
            _noise = tf.random.normal(shape=(param.batch_size,) + param.input_dim, mean=0.0,
                                      stddev=param.white_noise_std, dtype=tf.float32)
            _x_test = _x + _noise
        else:
            _x_test = _x

        _z_tilde = encoder(_x_test, training=False)
        _x_bar = decoder(_z_tilde, training=False)

        _loss_ae = tf.reduce_mean(tf.abs(tf.subtract(_x, _x_bar)))  # pixel-wise loss

        return _x_test, _x_bar, _loss_ae.numpy()
    ####################################################################################################################

    # 6. Inference
    train_mse = []
    for x_train, _ in train_set:
        x_noise, x_bar, loss_ae = testing_step(x_train)

        train_mse.append(loss_ae)

    num_test = 0
    valid_mse = []
    test_mse = []
    for x_test, _ in test_set:
        x_noise, x_bar, loss_ae = testing_step(x_test)

        if num_test <= param.valid_step:
            valid_mse.append(loss_ae)
        else:
            test_mse.append(loss_ae)
        num_test += 1

    # 7. Report
    print("[Loss] Train: {:.06f}\t Validation: {:.06f}\t Test: {:.06f}".format(np.mean(train_mse), np.mean(valid_mse),
                                                                               np.mean(test_mse)))

    # 8. Draw some samples
    if denoise is True:
        save_decode_image_array(x_noise.numpy(), path=os.path.join(graph_path, '{}_noise.png'.format(graph)))
    save_decode_image_array(x_test.numpy(), path=os.path.join(graph_path, '{}_original.png'.format(graph)))
    save_decode_image_array(x_bar.numpy(), path=os.path.join(graph_path, '{}_decoded.png'.format(graph)))


if __name__ == '__main__':
    inference(denoise=False)
    inference(denoise=True)
