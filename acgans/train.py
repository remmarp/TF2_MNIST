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
from acgans.parameter import Parameter
from acgans.networks import Generator, Discriminator, Classifier

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


################
#   DEFINITION #
################
def train():
    param = Parameter()

    # 1. Build models
    generator = Generator(param).model()
    discriminator = Discriminator(param).model()
    classifier = Classifier(param).model()

    # 2. Set optimizers
    opt_gen = tf.keras.optimizers.Adam(learning_rate=param.learning_rate_gen, beta_1=0.5, beta_2=0.999)
    opt_dis = tf.keras.optimizers.Adam(learning_rate=param.learning_rate_dis, beta_1=0.5, beta_2=0.999)
    opt_cla = tf.keras.optimizers.Adam(learning_rate=param.learning_rate_cla, beta_1=0.5, beta_2=0.999)

    # 3. Set trainable variables
    var_gen = generator.trainable_variables
    var_dis = discriminator.trainable_variables
    var_cla = generator.trainable_variables + classifier.trainable_variables + discriminator.trainable_variables[:-2]

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

    check_point_prefix = os.path.join(check_point_dir, 'acgans')

    gen_name = 'acgans_gen'
    dis_name = 'acgans_dis'
    cla_name = 'acgans_cla'

    graph = 'acgans'

    check_point = tf.train.Checkpoint(opt_gen=opt_gen, opt_dis=opt_dis, generator=generator, discriminator=discriminator,
                                      classifier=classifier)
    ckpt_manager = tf.train.CheckpointManager(check_point, check_point_dir, max_to_keep=5,
                                              checkpoint_name=check_point_prefix)

    # 7. Define train / validation step ################################################################################
    def training_step(_x, _y):
        with tf.GradientTape() as _gen_tape, tf.GradientTape() as _dis_tape, tf.GradientTape() as _cla_tape:
            _z = tf.random.uniform(shape=(param.batch_size, param.latent_dim), minval=-1.0, maxval=1.0,
                                   dtype=tf.float32)

            _gen_input = tf.concat([_y, _z], axis=-1, name='gen_input')
            _x_tilde = generator(_gen_input, training=True)

            _cla_real_logits, _dis_real = discriminator(_x, training=True)
            _cla_fake_logits, _dis_fake = discriminator(_x_tilde, training=True)

            _cla_real = classifier(_cla_real_logits, training=True)
            _cla_fake = classifier(_cla_fake_logits, training=True)

            _loss_cla_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(_y, axis=1),
                                                                                           logits=_cla_real))
            _loss_cla_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(_y, axis=1),
                                                                                           logits=_cla_fake))

            _real_loss, _fake_loss = -tf.reduce_mean(_dis_real), tf.reduce_mean(_dis_fake)
            _gp = gradient_penalty(partial(discriminator, training=True), _x, _x_tilde)

            _loss_cla = _loss_cla_real + _loss_cla_fake
            _loss_gen = -tf.reduce_mean(_dis_fake)
            _loss_dis = (_real_loss + _fake_loss) + _gp * param.w_gp_lambda

        _grad_gen = _gen_tape.gradient(_loss_gen, var_gen)
        _grad_dis = _dis_tape.gradient(_loss_dis, var_dis)
        _grad_cla = _cla_tape.gradient(_loss_cla, var_cla)

        opt_dis.apply_gradients(zip(_grad_dis, var_dis))
        opt_gen.apply_gradients(zip(_grad_gen, var_gen))
        opt_cla.apply_gradients(zip(_grad_cla, var_cla))

    def validation_step(_x, _y):
        _z = tf.random.uniform(shape=(param.batch_size, param.latent_dim), minval=-1.0, maxval=1.0,
                               dtype=tf.float32)

        _gen_input = tf.concat([_y, _z], axis=-1, name='gen_input')
        _x_tilde = generator(_gen_input, training=False)

        _cla_real_logits, _dis_real = discriminator(_x, training=False)
        _cla_fake_logits, _dis_fake = discriminator(_x_tilde, training=False)

        _cla_real = classifier(_cla_real_logits, training=False)
        _cla_fake = classifier(_cla_fake_logits, training=False)

        _loss_gen = -tf.reduce_mean(_dis_fake)

        _real_loss, _fake_loss = -tf.reduce_mean(_dis_real), tf.reduce_mean(_dis_fake)
        _gp = gradient_penalty(partial(discriminator, training=False), _x, _x_tilde)
        _loss_dis = (_real_loss + _fake_loss) + _gp * param.w_gp_lambda

        _loss_cla_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(_y, axis=1),
                                                                                       logits=_cla_real))
        _loss_cla_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(_y, axis=1),
                                                                                       logits=_cla_fake))
        _loss_cla = _loss_cla_real + _loss_cla_fake

        return _x_tilde, _loss_gen.numpy(), _loss_dis.numpy(), _loss_cla_real.numpy(), _loss_cla_fake.numpy(), (-_fake_loss.numpy()-_real_loss.numpy())
    ####################################################################################################################

    # 8. Train
    start_time = time.time()
    for epoch in range(0, param.max_epoch):
        # 8-1. Train ACGANs
        num_train = 0
        for x_train, y_train in train_set:
            training_step(x_train, tf.cast(y_train, dtype=tf.float32))
            num_train += 1

        # 8-2. Validation
        num_valid = 0
        val_loss_dis, val_loss_gen, val_loss_cla_real, val_loss_cla_fake, val_was_x = [], [], [], [], []
        for x_valid, y_valid in test_set:
            x_tilde, loss_dis, loss_gen, loss_cla_real, loss_cla_fake, was_x = validation_step(x_valid, tf.cast(y_valid, dtype=tf.float32))

            val_loss_dis.append(loss_dis)
            val_loss_gen.append(loss_gen)
            val_loss_cla_real.append(loss_cla_real)
            val_loss_cla_fake.append(loss_cla_fake)
            val_was_x.append(was_x)

            num_valid += 1

            if num_valid > param.valid_step:
                break

        # 8-3. Report in training
        elapsed_time = (time.time() - start_time) / 60.
        _val_loss_cla_real = np.mean(np.reshape(val_loss_cla_real, (-1)))
        _val_loss_cla_fake = np.mean(np.reshape(val_loss_cla_fake, (-1)))
        _val_loss_dis = np.mean(np.reshape(val_loss_dis, (-1)))
        _val_loss_gen = np.mean(np.reshape(val_loss_gen, (-1)))
        _val_was_x = np.mean(np.reshape(val_was_x, (-1)))

        print("[{:04d}] {:.01f} m.\tdis: {:.6f}\tgen: {:.6f}\tcla_r: {:.6f}\tcla_f: {:.6f}\t w_x: {:.4f}".format(epoch,
                                                                                                                 elapsed_time,
                                                                                                                 _val_loss_dis,
                                                                                                                 _val_loss_gen,
                                                                                                                 _val_loss_cla_real,
                                                                                                                 _val_loss_cla_fake,
                                                                                                                 _val_was_x))

        if epoch % param.save_frequency == 0 and epoch > 1:
            save_decode_image_array(x_valid.numpy(), path=os.path.join(graph_path,
                                                                       '{}_original-{:04d}.png'.format(graph,
                                                                                                       epoch)))
            save_decode_image_array(x_tilde.numpy(),
                                    path=os.path.join(graph_path, '{}_generated-{:04d}.png'.format(graph, epoch)))
            ckpt_manager.save(checkpoint_number=epoch)

    save_message = "\tSave model: End of training"

    generator.save_weights(os.path.join(model_path, gen_name))
    discriminator.save_weights(os.path.join(model_path, dis_name))
    classifier.save_weights(os.path.join(model_path, cla_name))

    print("[Epoch: {:04d}] {:.01f} min.".format(param.max_epoch, elapsed_time))
    print(save_message)


if __name__ == '__main__':
    train()
