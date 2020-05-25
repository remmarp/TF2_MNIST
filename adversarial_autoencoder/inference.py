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
from adversarial_autoencoder.networks import Encoder, Decoder, Discriminator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


################
#   DEFINITION #
################
def inference():
    param = Parameter()

    # 1. Build models
    encoder = Encoder(param).model()
    decoder = Decoder(param).model()
    discriminator = Discriminator(param).model()

    # 2. Load data
    data_loader = MNISTLoader(one_hot=True)
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
    enc_name = 'aae_enc'
    dec_name = 'aae_dec'
    dis_name = 'aae_dis'

    graph = 'aae'

    encoder.load_weights(os.path.join(model_path, enc_name))
    discriminator.load_weights(os.path.join(model_path, dis_name))
    decoder.load_weights(os.path.join(model_path, dec_name))

    # 5. Define loss

    # 6. Define testing step ###########################################################################################
    def testing_step(_x, _y):
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

        return _x_tilde, _x_bar, _loss_dis.numpy(), _loss_gen.numpy(), _loss_ae.numpy(), (
                    -_fake_loss.numpy() - _real_loss.numpy())
    ####################################################################################################################

    # 6. Inference
    train_loss_dis, train_loss_gen, train_loss_ae, train_was_x = [], [], [], []
    for x_train, y_train in train_set:
        y = tf.cast(y_train, dtype=tf.float32)
        x_tilde, x_bar, loss_dis, loss_gen, loss_ae, was_x = testing_step(x_train, y)

        train_loss_dis.append(loss_dis)
        train_loss_gen.append(loss_gen)
        train_loss_ae.append(loss_ae)
        train_was_x.append(was_x)

    num_test = 0
    val_loss_dis, val_loss_gen, val_loss_ae, val_was_x = [], [], [], []
    test_loss_dis, test_loss_gen, test_loss_ae, test_was_x = [], [], [], []

    for x_test, y_test in test_set:
        y = tf.cast(y_test, dtype=tf.float32)
        x_tilde, x_bar, loss_dis, loss_gen, loss_ae, was_x = testing_step(x_test, y)

        if num_test <= param.valid_step:
            val_loss_dis.append(loss_dis)
            val_loss_gen.append(loss_gen)
            val_loss_ae.append(loss_ae)
            val_was_x.append(was_x)
        else:
            test_loss_dis.append(loss_dis)
            test_loss_gen.append(loss_gen)
            test_loss_ae.append(loss_ae)
            test_was_x.append(was_x)

        num_test += 1

    _z = tf.random.normal(shape=(param.batch_size, param.latent_dim), mean=0.0,
                          stddev=param.prior_noise_std, dtype=tf.float32)
    for class_idx in range(0, param.num_class):
        _indices = np.ones(param.batch_size, dtype=np.float) * class_idx
        _y = tf.one_hot(indices=_indices, depth=param.num_class, dtype=tf.float32)

        _gen_input = tf.concat([_y, _z], axis=-1, name='gen_input')
        _x_tilde = decoder(_gen_input, training=False)

        save_decode_image_array(_x_tilde.numpy(), path=os.path.join(graph_path,
                                                                    '{}_c{}_generated.png'.format(graph, class_idx)))

    # 7. Report
    train_loss_dis = np.mean(np.reshape(train_loss_dis, (-1)))
    train_loss_gen = np.mean(np.reshape(train_loss_gen, (-1)))
    train_loss_ae = np.mean(np.reshape(train_loss_ae, (-1)))
    train_was_x = np.mean(np.reshape(train_was_x, (-1)))

    val_loss_dis = np.mean(np.reshape(val_loss_dis, (-1)))
    val_loss_gen = np.mean(np.reshape(val_loss_gen, (-1)))
    val_loss_ae = np.mean(np.reshape(val_loss_ae, (-1)))
    val_was_x = np.mean(np.reshape(val_was_x, (-1)))

    test_loss_dis = np.mean(np.reshape(test_loss_dis, (-1)))
    test_loss_gen = np.mean(np.reshape(test_loss_gen, (-1)))
    test_loss_ae = np.mean(np.reshape(test_loss_ae, (-1)))
    test_was_x = np.mean(np.reshape(test_was_x, (-1)))

    print("[Loss dis] Train: {:.06f}\t Validation: {:.06f}\t Test: {:.06f}".format(train_loss_dis, val_loss_dis,
                                                                                   test_loss_dis))
    print("[Loss gen] Train: {:.06f}\t Validation: {:.06f}\t Test: {:.06f}".format(train_loss_gen, val_loss_gen,
                                                                                   test_loss_gen))
    print("[Loss ae] Train: {:.06f}\t Validation: {:.06f}\t Test: {:.06f}".format(train_loss_ae, val_loss_ae,
                                                                                  test_loss_ae))
    print("[Was X] Train: {:.06f}\t Validation: {:.06f}\t Test: {:.06f}".format(train_was_x, val_was_x,
                                                                                test_was_x))

    # 8. Draw some samples
    save_decode_image_array(x_test.numpy(), path=os.path.join(graph_path, '{}_original.png'.format(graph)))
    save_decode_image_array(x_bar.numpy(), path=os.path.join(graph_path, '{}_decoded.png'.format(graph)))
    save_decode_image_array(x_tilde.numpy(), path=os.path.join(graph_path, '{}_generated.png'.format(graph)))


if __name__ == '__main__':
    inference()
