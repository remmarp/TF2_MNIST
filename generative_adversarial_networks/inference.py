# -*- coding: utf-8 -*-
# """
# generative_adversarial_networks/inference.py
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
from generative_adversarial_networks.parameter import Parameter
from generative_adversarial_networks.networks import Generator, Discriminator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


################
#   DEFINITION #
################
def inference(w_gp=False):
    param = Parameter()

    # 1. Build models
    generator = Generator(param).model()
    discriminator = Discriminator(param).model()

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
        gen_name = 'gan_w_gp'
        dis_name = 'dis_w_gp'
        graph = 'gan_w_gp'
    else:
        gen_name = 'gan'
        dis_name = 'dis'
        graph = 'gan'

    generator.load_weights(os.path.join(model_path, gen_name))
    discriminator.load_weights(os.path.join(model_path, dis_name))

    # 5. Define loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # 6. Inference
    train_dis_loss, train_gen_loss = [], []
    for x_train, _ in train_set:
        # noise = tf.random.uniform(shape=(param.batch_size, param.latent_dim), minval=-1, maxval=1, dtype=tf.float32)
        noise = tf.random.normal(shape=(param.batch_size, param.latent_dim), mean=0.0, stddev=0.3, dtype=tf.float32)
        x_tilde = generator(noise, training=False)

        dis_real = discriminator(x_train, training=False)
        dis_fake = discriminator(x_tilde, training=False)

        if w_gp is False:
            loss_dis = cross_entropy(tf.ones_like(dis_real), dis_real) + cross_entropy(tf.zeros_like(dis_fake),
                                                                                       dis_fake)
            loss_gen = cross_entropy(tf.ones_like(dis_fake), dis_fake)
        else:
            real_loss, fake_loss = -tf.reduce_mean(dis_real), tf.reduce_mean(dis_fake)
            gp = gradient_penalty(partial(discriminator, training=False), x_train, x_tilde)

            loss_dis = (real_loss + fake_loss) + gp * param.w_gp_lambda
            loss_gen = -tf.reduce_mean(dis_fake)

        train_dis_loss.append(loss_dis.numpy())
        train_gen_loss.append(loss_gen.numpy())

    num_test = 0
    valid_dis_loss, valid_gen_loss = [], []
    test_dis_loss, test_gen_loss = [], []

    for x_test, _ in test_set:
        # noise = tf.random.uniform(shape=(param.batch_size, param.latent_dim), minval=-1, maxval=1, dtype=tf.float32)
        noise = tf.random.normal(shape=(param.batch_size, param.latent_dim), mean=0.0, stddev=0.3, dtype=tf.float32)
        x_tilde = generator(noise, training=False)

        dis_real = discriminator(x_test, training=False)
        dis_fake = discriminator(x_tilde, training=False)

        if w_gp is False:
            loss_dis = cross_entropy(tf.ones_like(dis_real), dis_real) + cross_entropy(tf.zeros_like(dis_fake),
                                                                                       dis_fake)
            loss_gen = cross_entropy(tf.ones_like(dis_fake), dis_fake)
        else:
            real_loss, fake_loss = -tf.reduce_mean(dis_real), tf.reduce_mean(dis_fake)
            gp = gradient_penalty(partial(discriminator, training=False), x_test, x_tilde)

            loss_dis = (real_loss + fake_loss) + gp * param.w_gp_lambda
            loss_gen = -tf.reduce_mean(dis_fake)

        if num_test <= param.valid_step:
            valid_dis_loss.append(loss_dis.numpy())
            valid_gen_loss.append(loss_gen.numpy())

        else:
            test_dis_loss.append(loss_dis.numpy())
            test_gen_loss.append(loss_gen.numpy())
        num_test += 1

    # 7. Report
    train_dis_loss = np.mean(np.reshape(train_dis_loss, (-1)))
    valid_dis_loss = np.mean(np.reshape(valid_dis_loss, (-1)))
    test_dis_loss = np.mean(np.reshape(test_dis_loss, (-1)))

    train_gen_loss = np.mean(np.reshape(train_gen_loss, (-1)))
    valid_gen_loss = np.mean(np.reshape(valid_gen_loss, (-1)))
    test_gen_loss = np.mean(np.reshape(test_gen_loss, (-1)))

    print("[Loss dis] Train: {:.06f}\t Validation: {:.06f}\t Test: {:.06f}".format(train_dis_loss, valid_dis_loss,
                                                                                   test_dis_loss))
    print("[Loss gen] Train: {:.06f}\t Validation: {:.06f}\t Test: {:.06f}".format(train_gen_loss, valid_gen_loss,
                                                                                   test_gen_loss))

    # 8. Draw some samples
    save_decode_image_array(x_test.numpy(), path=os.path.join(graph_path, '{}_original.png'.format(graph)))
    save_decode_image_array(x_tilde.numpy(), path=os.path.join(graph_path, '{}_generated.png'.format(graph)))


if __name__ == '__main__':
    inference(False)
    inference(True)
