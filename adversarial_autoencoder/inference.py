# -*- coding: utf-8 -*-
# """
# adversarial_autoencoder/inference.py
# """

############
#   IMPORT #
############
# 1. Built-in modules
import os
import sys
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
from adversarial_autoencoder.networks import Encoder, Decoder, DiscriminatorZ

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


################
#   DEFINITION #
################
def inference(w_gp=False):
    param = Parameter()

    # 1. Build models
    encoder = Encoder(param).model()
    decoder = Decoder(param).model()
    discriminator_z = DiscriminatorZ(param).model()

    encoder.summary(line_length=param.model_display_len)
    decoder.summary(line_length=param.model_display_len)
    discriminator_z.summary(line_length=param.model_display_len)

    # 2. Load data
    data_loader = MNISTLoader(one_hot=False)
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
    if w_gp is True:
        enc_name = 'aae_enc_w_gp'
        dec_name = 'aae_dec_w_gp'
        dis_name = 'aae_dis_w_gp'

        graph = 'aae_w_gp'
    else:
        enc_name = 'aae_enc'
        dec_name = 'aae_dec'
        dis_name = 'aae_dis'

        graph = 'aae'

    encoder.load_weights(os.path.join(model_path, enc_name))
    decoder.load_weights(os.path.join(model_path, dec_name))
    discriminator_z.load_weights(os.path.join(model_path, dis_name))

    # 5. Define loss
    mse = tf.keras.losses.MeanSquaredError()
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # 6. Inference
    train_ae_loss, train_dis_loss, train_gen_loss = [], [], []
    for x_train, _ in train_set:
        z = tf.random.normal(shape=(param.batch_size, param.latent_dim), mean=0.0, stddev=0.3, dtype=tf.float32)
        z_tilde = encoder(x_train, training=False)
        x_bar = decoder(z_tilde, training=False)

        dis_real = discriminator_z(z, training=False)
        dis_fake = discriminator_z(z_tilde, training=False)

        if w_gp is False:
            loss_dis = tf.reduce_mean(cross_entropy(tf.ones_like(dis_real),
                                                    dis_real) + cross_entropy(tf.zeros_like(dis_fake),
                                                                              dis_fake))
            loss_gen = tf.reduce_mean(cross_entropy(tf.ones_like(dis_fake), dis_fake))
        else:
            real_loss, fake_loss = -tf.reduce_mean(dis_real), tf.reduce_mean(dis_fake)
            gp = gradient_penalty(partial(discriminator_z, training=False), z, z_tilde)

            loss_dis = (real_loss + fake_loss) + gp * param.w_gp_lambda
            loss_gen = -tf.reduce_mean(dis_fake)

        loss_ae = tf.reduce_mean(mse(x_train, x_bar))

        train_dis_loss.append(loss_dis.numpy())
        train_gen_loss.append(loss_gen.numpy())
        train_ae_loss.append(loss_ae.numpy())

    num_test = 0
    valid_ae_loss, valid_dis_loss, valid_gen_loss = [], [], []
    test_ae_loss, test_dis_loss, test_gen_loss = [], [], []

    for x_test, _ in test_set:
        z = tf.random.normal(shape=(param.batch_size, param.latent_dim), mean=0.0, stddev=0.3, dtype=tf.float32)
        z_tilde = encoder(x_test, training=False)
        x_bar = decoder(z_tilde, training=False)
        x_tilde = decoder(z, training=False)

        dis_real = discriminator_z(z, training=False)
        dis_fake = discriminator_z(z_tilde, training=False)

        if w_gp is False:
            loss_dis = tf.reduce_mean(cross_entropy(tf.ones_like(dis_real),
                                                    dis_real) + cross_entropy(tf.zeros_like(dis_fake),
                                                                              dis_fake))
            loss_gen = tf.reduce_mean(cross_entropy(tf.ones_like(dis_fake), dis_fake))
        else:
            real_loss, fake_loss = -tf.reduce_mean(dis_real), tf.reduce_mean(dis_fake)
            gp = gradient_penalty(partial(discriminator_z, training=False), z, z_tilde)

            loss_dis = (real_loss + fake_loss) + gp * param.w_gp_lambda
            loss_gen = -tf.reduce_mean(dis_fake)

        loss_ae = tf.reduce_mean(mse(x_test, x_bar))

        if num_test <= param.valid_step:
            valid_dis_loss.append(loss_dis.numpy())
            valid_gen_loss.append(loss_gen.numpy())
            valid_ae_loss.append(loss_ae.numpy())

        else:
            test_dis_loss.append(loss_dis.numpy())
            test_gen_loss.append(loss_gen.numpy())
            test_ae_loss.append(loss_ae.numpy())
        num_test += 1

    # 7. Report
    train_ae_loss = np.mean(np.reshape(train_ae_loss, (-1)))
    valid_ae_loss = np.mean(np.reshape(valid_ae_loss, (-1)))
    test_ae_loss = np.mean(np.reshape(test_ae_loss, (-1)))

    train_dis_loss = np.mean(np.reshape(train_dis_loss, (-1)))
    valid_dis_loss = np.mean(np.reshape(valid_dis_loss, (-1)))
    test_dis_loss = np.mean(np.reshape(test_dis_loss, (-1)))

    train_gen_loss = np.mean(np.reshape(train_gen_loss, (-1)))
    valid_gen_loss = np.mean(np.reshape(valid_gen_loss, (-1)))
    test_gen_loss = np.mean(np.reshape(test_gen_loss, (-1)))

    print("[Loss ae] Train: {:.06f}\t Validation: {:.06f}\t Test: {:.06f}".format(train_ae_loss, valid_ae_loss,
                                                                                   test_ae_loss))
    print("[Loss dis] Train: {:.06f}\t Validation: {:.06f}\t Test: {:.06f}".format(train_dis_loss, valid_dis_loss,
                                                                                   test_dis_loss))
    print("[Loss gen] Train: {:.06f}\t Validation: {:.06f}\t Test: {:.06f}".format(train_gen_loss, valid_gen_loss,
                                                                                   test_gen_loss))

    # 8. Draw some samples
    save_decode_image_array(x_test.numpy(), path=os.path.join(graph_path, '{}_original.png'.format(graph)))
    save_decode_image_array(x_bar.numpy(), path=os.path.join(graph_path, '{}_decode.png'.format(graph)))
    save_decode_image_array(x_tilde.numpy(), path=os.path.join(graph_path, '{}_generated.png'.format(graph)))


if __name__ == '__main__':
    inference(False)
    # inference(True)
