# -*- coding: utf-8 -*-
# """
# adversarial_autoencoder/train.py
# """

############
#   IMPORT #
############
# 1. Built-in modules
import os
import sys
import time
from functools import partial

sys.path.append(os.getcwd())

# 2. Third-party modules
import numpy as np
import tensorflow as tf

# 3. Own modules
from util import gradient_penalty
from data_loader import MNISTLoader
from visualize import save_decode_image_array
from adversarial_autoencoder.parameter import Parameter
from adversarial_autoencoder.networks import Encoder, Decoder, Discriminator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


################
#   DEFINITION #
################
def train():
    param = Parameter()

    # 1. Build models
    encoder = Encoder(param).model()
    decoder = Decoder(param).model()
    discriminator = Discriminator(param).model()

    # 2. Set optimizers
    opt_ae = tf.keras.optimizers.Adam(learning_rate=param.learning_rate_ae, beta_1=0.5, beta_2=0.999, epsilon=0.01)
    opt_gen = tf.keras.optimizers.Adam(learning_rate=param.learning_rate_gen, beta_1=0.5, beta_2=0.999, epsilon=0.01)
    opt_dis = tf.keras.optimizers.Adam(learning_rate=param.learning_rate_dis, beta_1=0.5, beta_2=0.999, epsilon=0.01)

    # 3. Set trainable variables
    var_ae = encoder.trainable_variables + decoder.trainable_variables
    var_gen = encoder.trainable_variables
    var_dis = discriminator.trainable_variables

    # 4. Load data
    data_loader = MNISTLoader(one_hot=True)
    train_set = data_loader.train.batch(batch_size=param.batch_size,
                                        drop_remainder=True).shuffle(buffer_size=data_loader.num_train,
                                                                     reshuffle_each_iteration=True)
    test_set = data_loader.test.batch(batch_size=param.batch_size, drop_remainder=True)

    # 5. Define loss

    # 6. Etc.
    check_point_dir = os.path.join(param.cur_dir, 'training_checkpoints')

    graph_path = os.path.join(param.cur_dir, 'graph')
    if not os.path.isdir(graph_path):
        os.mkdir(graph_path)

    model_path = os.path.join(param.cur_dir, 'model')
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    check_point_prefix = os.path.join(check_point_dir, 'aae')

    enc_name = 'aae_enc'
    dec_name = 'aae_dec'
    dis_name = 'aae_dis'

    graph = 'aae'

    check_point = tf.train.Checkpoint(opt_gen=opt_gen, opt_dis=opt_dis, opt_ae=opt_ae, encoder=encoder, decoder=decoder,
                                      discriminator=discriminator)
    ckpt_manager = tf.train.CheckpointManager(check_point, check_point_dir, max_to_keep=5,
                                              checkpoint_name=check_point_prefix)

    # 7. Define train / validation step ################################################################################
    def training_gen_step(_x, _y):
        with tf.GradientTape() as _gen_tape:
            _z_tilde = encoder(_x, training=True)

            _z_tilde_input = tf.concat([_y, _z_tilde], axis=-1, name='z_tilde_input')

            _dis_fake = discriminator(_z_tilde_input, training=True)

            _loss_gen = -tf.reduce_mean(_dis_fake)

        _grad_gen = _gen_tape.gradient(_loss_gen, var_gen)

        opt_gen.apply_gradients(zip(_grad_gen, var_gen))

    def training_step(_x, _y):
        with tf.GradientTape() as _ae_tape, tf.GradientTape() as _gen_tape, tf.GradientTape() as _dis_tape:
            _z = tf.random.normal(shape=(param.batch_size, param.latent_dim), mean=0.0,
                                  stddev=param.prior_noise_std, dtype=tf.float32)

            _z_tilde = encoder(_x, training=True)

            _z_input = tf.concat([_y, _z], axis=-1, name='z_input')
            _z_tilde_input = tf.concat([_y, _z_tilde], axis=-1, name='z_tilde_input')

            _x_bar = decoder(_z_tilde_input, training=True)

            _dis_real = discriminator(_z_input, training=True)
            _dis_fake = discriminator(_z_tilde_input, training=True)

            _real_loss, _fake_loss = -tf.reduce_mean(_dis_real), tf.reduce_mean(_dis_fake)
            _gp = gradient_penalty(partial(discriminator, training=True), _z_input, _z_tilde_input)

            _loss_gen = -tf.reduce_mean(_dis_fake)
            _loss_dis = (_real_loss + _fake_loss) + _gp * param.w_gp_lambda
            _loss_ae = tf.reduce_mean(tf.abs(tf.subtract(_x, _x_bar)))

        _grad_ae = _ae_tape.gradient(_loss_ae, var_ae)
        _grad_gen = _gen_tape.gradient(_loss_gen, var_gen)
        _grad_dis = _dis_tape.gradient(_loss_dis, var_dis)

        opt_ae.apply_gradients(zip(_grad_ae, var_ae))
        opt_gen.apply_gradients(zip(_grad_gen, var_gen))
        opt_dis.apply_gradients(zip(_grad_dis, var_dis))

    def validation_step(_x, _y):
        _z = tf.random.normal(shape=(param.batch_size, param.latent_dim), mean=0.0, stddev=param.prior_noise_std,
                              dtype=tf.float32)

        _z_tilde = encoder(_x, training=False)

        _z_input = tf.concat([_y, _z], axis=-1, name='z_input')
        _z_tilde_input = tf.concat([_y, _z_tilde], axis=-1, name='z_tilde_input')

        _x_bar = decoder(_z_tilde_input, training=False)
        _x_tilde = decoder(_z_input, training=False)

        _dis_real = discriminator(_z_input, training=False)
        _dis_fake = discriminator(_z_tilde_input, training=False)

        _real_loss, _fake_loss = -tf.reduce_mean(_dis_real), tf.reduce_mean(_dis_fake)
        _gp = gradient_penalty(partial(discriminator, training=False), _z_input, _z_tilde_input)

        _loss_gen = -tf.reduce_mean(_dis_fake)
        _loss_dis = (_real_loss + _fake_loss) + _gp * param.w_gp_lambda
        _loss_ae = tf.reduce_mean(tf.abs(tf.subtract(_x, _x_bar)))

        return _x_tilde, _x_bar, _loss_dis.numpy(), _loss_gen.numpy(), _loss_ae.numpy(), (-_fake_loss.numpy()-_real_loss.numpy())
    ####################################################################################################################

    # 8. Train
    start_time = time.time()
    for epoch in range(0, param.max_epoch):
        # 8-1. Train AAE
        num_train = 0
        for x_train, y_train in train_set:
            if num_train % 2 == 0:
                training_step(x_train, tf.cast(y_train, dtype=tf.float32))
            else:
                training_gen_step(x_train, tf.cast(y_train, dtype=tf.float32))
                training_gen_step(x_train, tf.cast(y_train, dtype=tf.float32))
            num_train += 1

        # 8-2. Validation
        num_valid = 0
        val_loss_dis, val_loss_gen, val_loss_ae, val_was_x = [], [], [], []
        for x_valid, y_valid in test_set:
            x_tilde, x_bar, loss_dis, loss_gen, loss_ae, was_x = validation_step(x_valid, tf.cast(y_valid,
                                                                                                  dtype=tf.float32))

            val_loss_dis.append(loss_dis)
            val_loss_gen.append(loss_gen)
            val_loss_ae.append(loss_ae)
            val_was_x.append(was_x)

            num_valid += 1

            if num_valid > param.valid_step:
                break

        # 8-3. Report in training
        elapsed_time = (time.time() - start_time) / 60.
        _val_loss_ae = np.mean(np.reshape(val_loss_ae, (-1)))
        _val_loss_dis = np.mean(np.reshape(val_loss_dis, (-1)))
        _val_loss_gen = np.mean(np.reshape(val_loss_gen, (-1)))
        _val_was_x = np.mean(np.reshape(val_was_x, (-1)))

        print("[Epoch: {:04d}] {:.01f}m.\tdis: {:.6f}\tgen: {:.6f}\tae: {:.6f}\tw_x: {:.6f}".format(epoch,
                                                                                                    elapsed_time,
                                                                                                    _val_loss_dis,
                                                                                                    _val_loss_gen,
                                                                                                    _val_loss_ae,
                                                                                                    _val_was_x))

        if epoch % param.save_frequency == 0 and epoch > 1:
            save_decode_image_array(x_valid.numpy(), path=os.path.join(graph_path,
                                                                       '{}_original-{:04d}.png'.format(graph,
                                                                                                       epoch)))
            save_decode_image_array(x_bar.numpy(), path=os.path.join(graph_path,
                                                                     '{}_decode-{:04d}.png'.format(graph, epoch)))
            save_decode_image_array(x_tilde.numpy(),
                                    path=os.path.join(graph_path, '{}_generated-{:04d}.png'.format(graph, epoch)))
            ckpt_manager.save(checkpoint_number=epoch)

    save_message = "\tSave model: End of training"

    encoder.save_weights(os.path.join(model_path, enc_name))
    decoder.save_weights(os.path.join(model_path, dec_name))
    discriminator.save_weights(os.path.join(model_path, dis_name))

    # 6-3. Report
    print("[Epoch: {:04d}] {:.01f} min.".format(param.max_epoch, elapsed_time))
    print(save_message)


if __name__ == '__main__':
    train()
