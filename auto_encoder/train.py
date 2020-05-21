# -*- coding: utf-8 -*-
# """
# auto_encoder/train.py
# """

############
#   IMPORT #
############
# 1. Built-in modules
import os
import sys
import time

# 2. Third-party modules
import numpy as np
import tensorflow as tf

# 3. Own modules
sys.path.append(os.getcwd())
from data_loader import MNISTLoader
from auto_encoder.parameter import Parameter
from visualize import save_decode_image_array
from auto_encoder.networks import Encoder, Decoder

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


################
#   DEFINITION #
################
def train(denoise=False):
    param = Parameter()

    # 1. Build models
    encoder = Encoder(param).model()
    decoder = Decoder(param).model()

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

    # 6. Etc.
    check_point_dir = os.path.join(param.cur_dir, 'training_checkpoints')

    model_path = os.path.join(param.cur_dir, 'model')
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    graph_path = os.path.join(param.cur_dir, 'graph')
    if not os.path.isdir(graph_path):
        os.mkdir(graph_path)

    minimum_mse = 100.
    num_effective_epoch = 0

    if denoise is True:
        check_point_prefix = os.path.join(check_point_dir, 'ae_denoise')
        graph = 'ae_denoise'
        enc_name = 'encoder_denoise'
        dec_name = 'decoder_denoise'
    else:
        check_point_prefix = os.path.join(check_point_dir, 'ae')
        graph = 'ae'
        enc_name = 'encoder'
        dec_name = 'decoder'

    check_point = tf.train.Checkpoint(opt_ae=opt_ae, encoder=encoder, decoder=decoder)
    ckpt_manager = tf.train.CheckpointManager(check_point, check_point_dir, max_to_keep=5,
                                              checkpoint_name=check_point_prefix)

    # 7. Define train / validation step ################################################################################
    def training_step(_x):
        if denoise is True:
            _noise = tf.random.normal(shape=(param.batch_size,) + param.input_dim, mean=0.0,
                                      stddev=param.white_noise_std, dtype=tf.float32)
            _x_train = _x + _noise
        else:
            _x_train = _x

        with tf.GradientTape() as _ae_tape:
            _z_tilde = encoder(_x_train, training=True)
            _x_bar = decoder(_z_tilde, training=True)

            _loss_ae = tf.reduce_mean(tf.abs(tf.subtract(_x, _x_bar)))  # pixel-wise loss

        _grad_ae = _ae_tape.gradient(_loss_ae, var_ae)
        opt_ae.apply_gradients(zip(_grad_ae, var_ae))

    def validation_step(_x):
        if denoise is True:
            _noise = tf.random.normal(shape=(param.batch_size,) + param.input_dim, mean=0.0,
                                      stddev=param.white_noise_std, dtype=tf.float32)
            _x_valid = _x + _noise
        else:
            _x_valid = _x

        _z_tilde = encoder(_x_valid, training=False)
        _x_bar = decoder(_z_tilde, training=False)

        _loss_ae = tf.reduce_mean(tf.abs(tf.subtract(_x, _x_bar)))  # pixel-wise loss

        return _x_valid, _x_bar, _loss_ae.numpy()
    ####################################################################################################################

    # 8. Train
    start_time = time.time()
    for epoch in range(0, param.max_epoch):
        # 8-1. Train auto encoder
        for x_train, _ in train_set:
            training_step(x_train)

        # 8-2. Validation
        num_valid = 0
        mse_valid = []
        for x_valid, _ in test_set:
            x_noise, x_bar, loss_ae = validation_step(x_valid)
            mse_valid.append(loss_ae)

            if num_valid == param.valid_step:
                break
            num_valid += 1

        valid_loss = np.mean(mse_valid)

        save_message = ''
        if minimum_mse > valid_loss:
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

        if epoch % param.save_frequency == 0 and epoch > 1:
            if denoise is True:
                save_decode_image_array(x_noise.numpy(), path=os.path.join(graph_path,
                                                                           '{}_noise_{:04d}.png'.format(graph, epoch)))
            save_decode_image_array(x_valid.numpy(), path=os.path.join(graph_path,
                                                                       '{}_original_{:04d}.png'.format(graph, epoch)))
            save_decode_image_array(x_bar.numpy(), path=os.path.join(graph_path, '{}_decoded_{:04d}.png'.format(graph,
                                                                                                                epoch)))
            ckpt_manager.save(checkpoint_number=epoch)

        if num_effective_epoch >= param.num_early_stopping:
            print("\t Early stopping at epoch {:04d}!".format(epoch))
            break

        num_effective_epoch += 1
    print("\t Stopping at epoch {:04d}!".format(epoch))


if __name__ == '__main__':
    train(denoise=False)
    train(denoise=True)
