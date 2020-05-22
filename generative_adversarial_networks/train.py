# -*- coding: utf-8 -*-
# """
# generative_adversarial_networks/train.py
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
from generative_adversarial_networks.parameter import Parameter
from generative_adversarial_networks.networks import Generator, Discriminator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


################
#   DEFINITION #
################
def train(w_gp=False):
    param = Parameter()

    # 1. Build models
    generator = Generator(param).model()
    discriminator = Discriminator(param).model()

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
    check_point_dir = os.path.join(param.cur_dir, 'training_checkpoints')

    graph_path = os.path.join(param.cur_dir, 'graph')
    if not os.path.isdir(graph_path):
        os.mkdir(graph_path)

    model_path = os.path.join(param.cur_dir, 'model')
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    if w_gp is True:
        check_point_prefix = os.path.join(check_point_dir, 'gan_w_gp')
        gen_name = 'gan_w_gp'
        dis_name = 'dis_w_gp'
        graph = 'gan_w_gp'
    else:
        check_point_prefix = os.path.join(check_point_dir, 'gan')
        gen_name = 'gan'
        dis_name = 'dis'
        graph = 'gan'

    check_point = tf.train.Checkpoint(opt_gen=opt_gen, opt_dis=opt_dis, generator=generator,
                                      discriminator=discriminator)
    ckpt_manager = tf.train.CheckpointManager(check_point, check_point_dir, max_to_keep=5,
                                              checkpoint_name=check_point_prefix)

    # 7. Define train / validation step ################################################################################
    def training_step(_x):
        with tf.GradientTape() as _gen_tape, tf.GradientTape() as _dis_tape:
            _noise = tf.random.normal(shape=(param.batch_size, param.latent_dim), mean=0.0,
                                      stddev=param.prior_noise_std, dtype=tf.float32)

            _x_tilde = generator(_noise, training=True)

            _dis_real = discriminator(_x, training=True)
            _dis_fake = discriminator(_x_tilde, training=True)

            if w_gp is True:
                _real_loss, _fake_loss = -tf.reduce_mean(_dis_real), tf.reduce_mean(_dis_fake)
                _gp = gradient_penalty(partial(discriminator, training=True), _x, _x_tilde)

                _loss_dis = (_real_loss + _fake_loss) + _gp * param.w_gp_lambda
                _loss_gen = -tf.reduce_mean(_dis_fake)
            else:
                _loss_dis = cross_entropy(tf.ones_like(_dis_real), _dis_real) + cross_entropy(tf.zeros_like(_dis_fake),
                                                                                              _dis_fake)
                _loss_gen = cross_entropy(tf.ones_like(_dis_fake), _dis_fake)

        _grad_gen = _gen_tape.gradient(_loss_gen, var_gen)
        _grad_dis = _dis_tape.gradient(_loss_dis, var_dis)

        opt_gen.apply_gradients(zip(_grad_gen, var_gen))
        opt_dis.apply_gradients(zip(_grad_dis, var_dis))

    def validation_step(_x):
        _noise = tf.random.normal(shape=(param.batch_size, param.latent_dim), mean=0.0,
                                  stddev=param.prior_noise_std, dtype=tf.float32)

        _x_tilde = generator(_noise, training=False)

        _dis_real = discriminator(_x, training=False)
        _dis_fake = discriminator(_x_tilde, training=False)

        if w_gp is True:
            _real_loss, _fake_loss = -tf.reduce_mean(_dis_real), tf.reduce_mean(_dis_fake)
            _gp = gradient_penalty(partial(discriminator, training=False), _x, _x_tilde)

            _loss_dis = (_real_loss + _fake_loss) + _gp * param.w_gp_lambda
            _loss_gen = -tf.reduce_mean(_dis_fake)
        else:
            _loss_dis = cross_entropy(tf.ones_like(_dis_real), _dis_real) + cross_entropy(tf.zeros_like(_dis_fake),
                                                                                          _dis_fake)
            _loss_gen = cross_entropy(tf.ones_like(_dis_fake), _dis_fake)

        return _x_tilde, _loss_dis.numpy(), _loss_gen.numpy()
    ####################################################################################################################

    # 8. Train
    start_time = time.time()
    for epoch in range(0, param.max_epoch):
        # 8-1. Train GANs
        for x_train, _ in train_set:
            training_step(x_train)

        # 8-2. Validation
        num_valid = 0
        val_loss_dis, val_loss_gen = [], []
        for x_valid, _ in test_set:
            if num_valid == param.valid_step:
                break

            x_tilde, loss_dis, loss_gen = validation_step(x_valid)

            val_loss_dis.append(loss_dis)
            val_loss_gen.append(loss_gen)

            num_valid += 1

        if epoch % param.save_frequency == 0 and epoch > 1:
            save_decode_image_array(x_valid.numpy(), path=os.path.join(graph_path,
                                                                       '{}_original-{:04d}.png'.format(graph,
                                                                                                       epoch)))
            save_decode_image_array(x_tilde.numpy(),
                                    path=os.path.join(graph_path, '{}_generated-{:04d}.png'.format(graph, epoch)))

            ckpt_manager.save(checkpoint_number=epoch)

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
