# -*- coding: utf-8 -*-
# """
# auto_encoder/train.py
# """

############
#   IMPORT #
############
# 1. Built-in modules
import os
import time

# 2. Third-party modules
import numpy as np
import tensorflow as tf

# 3. Own modules
from data_loader import MNISTLoader
from auto_encoder.parameter import Parameter
from auto_encoder.networks import Encoder, Decoder


################
#   DEFINITION #
################
def train(denoise=False):
    param = Parameter()

    # 1. Build models
    encoder = Encoder(param).model()
    decoder = Decoder(param).model()

    encoder.summary(line_length=param.model_display_len)
    decoder.summary(line_length=param.model_display_len)

    # 2. Set optimizers
    opt_ae = tf.keras.optimizers.Adam(learning_rate=param.learning_rate_ae)

    # 3. Set trainable variables
    var_ae = encoder.trainable_variables + decoder.trainable_variables

    # 4. Load data
    data_loader = MNISTLoader()
    train_set = data_loader.train.batch(batch_size=param.batch_size,
                                        drop_remainder=True).shuffle(buffer_size=data_loader.num_train,
                                                                     reshuffle_each_iteration=True)
    test_set = data_loader.test.batch(batch_size=param.batch_size, drop_remainder=True)

    # 5. Define loss
    mse = tf.keras.losses.MeanSquaredError()

    # 6. Etc.
    graph_path = os.path.join(os.getcwd(), 'graph')
    if not os.path.isdir(graph_path):
        os.mkdir(graph_path)

    model_path = os.path.join(os.getcwd(), 'model')
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    minimum_mse = 100.
    num_effective_epoch = 0

    if denoise is True:
        enc_name = 'mse_denoise_ae_encoder'
        dec_name = 'mse_denoise_ae_decoder'
    else:
        enc_name = 'mse_ae_encoder'
        dec_name = 'mse_ae_decoder'

    # 7. Train
    start_time = time.time()
    for epoch in range(0, param.max_epoch):
        # 7-1. Train auto encoder
        for x_train, _ in train_set:
            with tf.GradientTape() as ae_tape:
                if denoise is True:
                    noise = tf.random.normal(shape=(param.batch_size,)+param.input_dim, mean=0.0,
                                             stddev=param.white_noise_std, dtype=tf.float32)
                    x = x_train + noise
                else:
                    x = x_train

                z = encoder(x, training=True)
                x_bar = decoder(z, training=True)

                loss_ae = tf.reduce_mean(mse(x_train, x_bar))

            grad_ae = ae_tape.gradient(loss_ae, sources=var_ae)
            opt_ae.apply_gradients(zip(grad_ae, var_ae))

        # 7-2. Validation
        num_valid = 0
        mse_valid = []
        for x_valid, _ in test_set:
            z = encoder(x_valid, training=False)
            x_bar = decoder(z, training=False)

            loss_ae = tf.reduce_mean(mse(x_valid, x_bar))
            mse_valid.append(loss_ae.numpy())

            if num_valid == param.valid_step:
                break
            num_valid += 1

        valid_loss = np.mean(mse_valid)

        save_message = ''
        if minimum_mse > np.mean(mse_valid):
            num_effective_epoch = 0
            minimum_mse = valid_loss
            save_message = "\tSave model: detecting lowest mse: {:.6f} at epoch {:04d}".format(minimum_mse, epoch)

            encoder.save_weights(os.path.join(model_path, enc_name))
            decoder.save_weights(os.path.join(model_path, dec_name))

        elapsed_time = (time.time() - start_time) / 60.

        # 7-3. Report
        print("[Epoch: {:04d}] {:.01f} min. ae loss: {:.6f} Effective: {}".format(epoch, elapsed_time,
                                                                                  valid_loss,
                                                                                  (num_effective_epoch == 0)))
        print("{}".format(save_message))
        if num_effective_epoch >= param.num_early_stopping:
            print("\t Early stopping at epoch {:04d}!".format(epoch))
            exit(0)

        num_effective_epoch += 1


if __name__ == '__main__':
    train(denoise=False)
    train(denoise=True)
