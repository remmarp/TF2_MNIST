# -*- coding: utf-8 -*-
# """
# generative_adversarial_networks/train.py
# """

############
#   IMPORT #
############
# 1. Built-in modules
import os
import time
from functools import partial

# 2. Third-party modules
import numpy as np
import tensorflow as tf

# 3. Own modules
from util import gradient_penalty
from data_loader import MNISTLoader
from generative_adversarial_networks.parameter import Parameter
from generative_adversarial_networks.networks import Generator, Discriminator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


################
#   DEFINITION #
################
def train(w_gp=False):
    param = Parameter()

    # 1. Build models
    generator = Generator(param).model()
    discriminator = Discriminator(param).model()

    generator.summary(line_length=param.model_display_len)
    discriminator.summary(line_length=param.model_display_len)

    # 2. Set optimizers
    opt_gen = tf.keras.optimizers.Adam(learning_rate=param.learning_rate_gen)
    opt_dis = tf.keras.optimizers.Adam(learning_rate=param.learning_rate_dis)

    # 3. Set trainable variables
    var_gen = generator.trainable_variables
    var_dis = discriminator.trainable_variables

    # 4. Load data
    data_loader = MNISTLoader(one_hot=False)
    train_set = data_loader.train.batch(batch_size=param.batch_size,
                                        drop_remainder=True).shuffle(buffer_size=data_loader.num_train,
                                                                     reshuffle_each_iteration=True)
    test_set = data_loader.test.batch(batch_size=param.batch_size, drop_remainder=True)

    # 5. Define loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    # 6. Etc.
    graph_path = os.path.join(os.getcwd(), 'graph')
    if not os.path.isdir(graph_path):
        os.mkdir(graph_path)

    model_path = os.path.join(os.getcwd(), 'model')
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    if w_gp is True:
        gen_name = 'gan_w_gp'
        dis_name = 'dis_w_gp'
    else:
        gen_name = 'gan'
        dis_name = 'dis'


    # 7. Train
    start_time = time.time()
    for epoch in range(0, param.max_epoch):
        # 7-1. Train GANs
        for x_train, _ in train_set:
            noise = tf.random.normal(shape=(param.batch_size, param.latent_dim), mean=0.0, stddev=0.3, dtype=tf.float32)
            # noise = tf.random.uniform(shape=(param.batch_size, param.latent_dim), minval=-1, maxval=1, dtype=tf.float32)
            with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
                x_tilde = generator(noise, training=True)

                dis_real = discriminator(x_train, training=True)
                dis_fake = discriminator(x_tilde, training=True)

                if w_gp is False:
                    loss_dis = cross_entropy(tf.ones_like(dis_real), dis_real) + cross_entropy(tf.zeros_like(dis_fake),
                                                                                               dis_fake)
                    loss_gen = cross_entropy(tf.ones_like(dis_fake), dis_fake)
                else:
                    real_loss, fake_loss = -tf.reduce_mean(dis_real), tf.reduce_mean(dis_fake)
                    gp = gradient_penalty(partial(discriminator, training=True), x_train, x_tilde)

                    loss_dis = (real_loss + fake_loss) + gp * param.w_gp_lambda
                    loss_gen = -tf.reduce_mean(dis_fake)

            grad_gen = gen_tape.gradient(loss_gen, var_gen)
            grad_dis = dis_tape.gradient(loss_dis, var_dis)

            opt_gen.apply_gradients(zip(grad_gen, var_gen))
            opt_dis.apply_gradients(zip(grad_dis, var_dis))

        # 7-2. Validation
        num_valid = 0
        val_loss_dis, val_loss_gen = [], []
        for x_valid, _ in test_set:
            if num_valid == param.valid_step:
                break
            noise = tf.random.normal(shape=(param.batch_size, param.latent_dim), mean=0.0, stddev=0.3, dtype=tf.float32)
            # noise = tf.random.uniform(shape=(param.batch_size, param.latent_dim), minval=-1, maxval=1, dtype=tf.float32)
            x_tilde = generator(noise, training=False)

            dis_real = discriminator(x_valid, training=False)
            dis_fake = discriminator(x_tilde, training=False)

            if w_gp is False:
                loss_dis = cross_entropy(tf.ones_like(dis_real), dis_real) + cross_entropy(tf.zeros_like(dis_fake),
                                                                                           dis_fake)
                loss_gen = cross_entropy(tf.ones_like(dis_fake), dis_fake)
            else:
                real_loss, fake_loss = -tf.reduce_mean(dis_real), tf.reduce_mean(dis_fake)
                gp = gradient_penalty(partial(discriminator, training=False), x_valid, x_tilde)

                loss_dis = (real_loss + fake_loss) + gp * param.w_gp_lambda
                loss_gen = -tf.reduce_mean(dis_fake)

            val_loss_dis.append(loss_dis.numpy())
            val_loss_gen.append(loss_gen.numpy())

            num_valid += 1

        # 7-3. Report in training
        elapsed_time = (time.time() - start_time) / 60.
        _val_loss_dis = np.mean(np.reshape(val_loss_dis, (-1)))
        _val_loss_gen = np.mean(np.reshape(val_loss_gen, (-1)))
        print("[Epoch: {:04d}] {:.01f} min.\t loss dis: {:.6f}\t loss gen: {:.6f}".format(epoch, elapsed_time,
                                                                                          _val_loss_dis,
                                                                                          _val_loss_gen))

    save_message = "\tSave model: End of training"

    generator.save_weights(os.path.join(model_path, gen_name))
    discriminator.save_weights(os.path.join(model_path, dis_name))

    # 6-3. Report
    print("[Epoch: {:04d}] {:.01f} min.".format(param.max_epoch, elapsed_time))
    print(save_message)


if __name__ == '__main__':
    train(False)
    train(True)
