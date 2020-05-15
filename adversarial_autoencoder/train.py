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
from adversarial_autoencoder.parameter import Parameter
from adversarial_autoencoder.networks import Encoder, Decoder, DiscriminatorZ

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


################
#   DEFINITION #
################
def train(w_gp=False):
    param = Parameter()

    # 1. Build models
    encoder = Encoder(param).model()
    decoder = Decoder(param).model()
    discriminator_z = DiscriminatorZ(param).model()

    encoder.summary(line_length=param.model_display_len)
    decoder.summary(line_length=param.model_display_len)
    discriminator_z.summary(line_length=param.model_display_len)

    # 2. Set optimizers
    opt_ae = tf.keras.optimizers.Adam(learning_rate=param.learning_rate_ae)
    opt_gen = tf.keras.optimizers.Adam(learning_rate=param.learning_rate_gen)
    opt_dis = tf.keras.optimizers.Adam(learning_rate=param.learning_rate_dis)

    # 3. Set trainable variables
    var_ae = encoder.trainable_variables + decoder.trainable_variables
    var_gen = encoder.trainable_variables
    var_dis = discriminator_z.trainable_variables

    # 4. Load data
    data_loader = MNISTLoader(one_hot=False)
    train_set = data_loader.train.batch(batch_size=param.batch_size,
                                        drop_remainder=True).shuffle(buffer_size=data_loader.num_train,
                                                                     reshuffle_each_iteration=True)
    test_set = data_loader.test.batch(batch_size=param.batch_size, drop_remainder=True)

    # 5. Define loss
    mse = tf.keras.losses.MeanSquaredError()
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # 6. Etc.
    graph_path = os.path.join(param.cur_dir, 'graph')
    if not os.path.isdir(graph_path):
        os.mkdir(graph_path)

    model_path = os.path.join(param.cur_dir, 'model')
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    if w_gp is True:
        enc_name = 'aae_enc_w_gp'
        dec_name = 'aae_dec_w_gp'
        dis_name = 'aae_dis_w_gp'
    else:
        enc_name = 'aae_enc'
        dec_name = 'aae_dec'
        dis_name = 'aae_dis'

    # 7. Train
    start_time = time.time()
    for epoch in range(0, param.max_epoch):
        # 7-1. Pre-train auto-encoder
        if epoch < param.pre_train_epoch:
            for x_train, _ in train_set:
                with tf.GradientTape() as ae_tape:
                    z_tilde = encoder(x_train, training=True)
                    x_bar = decoder(z_tilde, training=True)

                    loss_ae = tf.reduce_mean(mse(x_train, x_bar))
                grad_ae = ae_tape.gradient(loss_ae, sources=var_ae)
                opt_ae.apply_gradients(zip(grad_ae, var_ae))

        # 7-2. Train AAE
        else:
            num_train = 0
            for x_train, _ in train_set:
                with tf.GradientTape() as ae_tape, tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
                    z_tilde = encoder(x_train, training=True)  # generate

                    z = tf.random.normal(shape=(param.batch_size, param.latent_dim), mean=0.0, stddev=0.3,
                                         dtype=tf.float32)
                    x_bar = decoder(z_tilde, training=True)  # decode

                    dis_fake = discriminator_z(z_tilde, training=True)  # Discriminate

                    loss_ae = tf.reduce_mean(mse(x_train, x_bar))

                    if w_gp is False:
                        loss_gen = tf.reduce_mean(cross_entropy(tf.ones_like(dis_fake), dis_fake))


                        dis_real = discriminator_z(z, training=True)  # Discriminate

                        loss_dis = tf.reduce_mean(cross_entropy(tf.ones_like(dis_real),
                                                                dis_real) + cross_entropy(tf.zeros_like(dis_fake),
                                                                                          dis_fake))

                    else:
                        loss_gen = -tf.reduce_mean(dis_fake)
                        real_loss, fake_loss = -tf.reduce_mean(dis_real), tf.reduce_mean(dis_fake)
                        gp = gradient_penalty(partial(discriminator_z, training=True), z, z_tilde)

                        loss_dis = (real_loss + fake_loss) + gp * param.w_gp_lambda

                grad_gen = gen_tape.gradient(loss_gen, var_gen)
                grad_ae = ae_tape.gradient(loss_ae, var_ae)
                grad_dis = dis_tape.gradient(loss_dis, var_dis)

                opt_ae.apply_gradients(zip(grad_ae, var_ae))
                opt_gen.apply_gradients(zip(grad_gen, var_gen))
                opt_dis.apply_gradients(zip(grad_dis, var_dis))

                num_train += 1

        # 7-2. Validation
        num_valid = 0
        val_loss_dis, val_loss_gen, val_loss_ae = [], [], []
        for x_valid, _ in test_set:
            if num_valid == param.valid_step:
                break
            z = tf.random.normal(shape=(param.batch_size, param.latent_dim), mean=0.0, stddev=0.3, dtype=tf.float32)
            z_tilde = encoder(x_valid, training=False)
            x_bar = decoder(z_tilde, training=False)

            dis_real = discriminator_z(z, training=False)
            dis_fake = discriminator_z(z_tilde, training=False)

            if w_gp is False:
                loss_dis = tf.reduce_mean(
                    cross_entropy(tf.ones_like(dis_real), dis_real) + cross_entropy(tf.zeros_like(dis_fake),
                                                                                    dis_fake))
                loss_gen = tf.reduce_mean(cross_entropy(tf.ones_like(dis_fake), dis_fake))
            else:
                real_loss, fake_loss = -tf.reduce_mean(dis_real), tf.reduce_mean(dis_fake)
                gp = gradient_penalty(partial(discriminator_z, training=False), z, z_tilde)

                loss_dis = (real_loss + fake_loss) + gp * param.w_gp_lambda
                loss_gen = -tf.reduce_mean(dis_fake)

            loss_ae = tf.reduce_mean(mse(x_valid, x_bar))

            val_loss_dis.append(loss_dis.numpy())
            val_loss_gen.append(loss_gen.numpy())
            val_loss_ae.append(loss_ae.numpy())

            num_valid += 1

        # 7-3. Report in training
        elapsed_time = (time.time() - start_time) / 60.
        _val_loss_ae = np.mean(np.reshape(val_loss_ae, (-1)))
        _val_loss_dis = np.mean(np.reshape(val_loss_dis, (-1)))
        _val_loss_gen = np.mean(np.reshape(val_loss_gen, (-1)))

        if epoch < param.pre_train_epoch:
            print("[Epoch: {:04d}] {:.01f} min.\t Pre-training... loss ae: {:.6f}".format(epoch, elapsed_time,
                                                                                          _val_loss_ae))
        else:
            print("[Epoch: {:04d}] {:.01f} min.\t loss dis: {:.6f}\t loss gen: {:.6f}\t loss ae: {:.6f}".format(epoch,
                                                                                                                elapsed_time,
                                                                                                                _val_loss_dis,
                                                                                                                _val_loss_gen,
                                                                                                                _val_loss_ae))

    save_message = "\tSave model: End of training"

    encoder.save_weights(os.path.join(model_path, enc_name))
    decoder.save_weights(os.path.join(model_path, dec_name))
    discriminator_z.save_weights(os.path.join(model_path, dis_name))

    # 6-3. Report
    print("[Epoch: {:04d}] {:.01f} min.".format(param.max_epoch, elapsed_time))
    print(save_message)


if __name__ == '__main__':
    train(False)
