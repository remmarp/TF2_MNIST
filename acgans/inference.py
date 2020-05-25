# -*- coding: utf-8 -*-
# """
# acgans/inference.py
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
import sklearn.metrics as metrics

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
def inference():
    param = Parameter()

    # 1. Build models
    generator = Generator(param).model()
    discriminator = Discriminator(param).model()
    classifier = Classifier(param).model()

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
    gen_name = 'acgans_gen'
    dis_name = 'acgans_dis'
    cla_name = 'acgans_cla'

    graph = 'acgans'

    generator.load_weights(os.path.join(model_path, gen_name))
    discriminator.load_weights(os.path.join(model_path, dis_name))
    classifier.load_weights(os.path.join(model_path, cla_name))

    # 5. Define loss

    # 6. Define testing step ###########################################################################################
    def testing_step(_x, _y):
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

        return _x_tilde, _cla_real, _loss_gen.numpy(), _loss_dis.numpy(), _loss_cla_real.numpy(), _loss_cla_fake.numpy(), (-_fake_loss.numpy()-_real_loss.numpy())
    ####################################################################################################################

    # 6. Inference
    train_real_label, train_prediction = [], []
    train_loss_dis, train_loss_gen, train_loss_cla_real, train_loss_cla_fake, train_was_x = [], [], [], [], []
    for x_train, y_train in train_set:
        y = tf.cast(y_train, dtype=tf.float32)
        x_tilde, prediction, loss_dis, loss_gen, loss_cla_real, loss_cla_fake, was_x = testing_step(x_train, y)

        _pred_y = tf.argmax(prediction, axis=1)

        train_loss_dis.append(loss_dis)
        train_loss_gen.append(loss_gen)
        train_loss_cla_real.append(loss_cla_real)
        train_loss_cla_fake.append(loss_cla_fake)
        train_was_x.append(was_x)

        train_real_label.append(tf.argmax(y_train, axis=1).numpy())
        train_prediction.append(_pred_y)

    num_test = 0
    valid_real_label, valid_prediction = [], []
    test_real_label, test_prediction = [], []
    val_loss_dis, val_loss_gen, val_loss_cla_real, val_loss_cla_fake, val_was_x = [], [], [], [], []
    test_loss_dis, test_loss_gen, test_loss_cla_real, test_loss_cla_fake, test_was_x = [], [], [], [], []

    for x_test, y_test in test_set:
        y = tf.cast(y_test, dtype=tf.float32)
        x_tilde, prediction, loss_dis, loss_gen, loss_cla_real, loss_cla_fake, was_x = testing_step(x_test, y)

        _pred_y = tf.argmax(prediction, axis=1)

        if num_test <= param.valid_step:
            val_loss_dis.append(loss_dis)
            val_loss_gen.append(loss_gen)
            val_loss_cla_real.append(loss_cla_real)
            val_loss_cla_fake.append(loss_cla_fake)
            val_was_x.append(was_x)

            valid_real_label.append(tf.argmax(y_test, axis=1).numpy())
            valid_prediction.append(_pred_y)

        else:
            test_loss_dis.append(loss_dis)
            test_loss_gen.append(loss_gen)
            test_loss_cla_real.append(loss_cla_real)
            test_loss_cla_fake.append(loss_cla_fake)
            test_was_x.append(was_x)

            test_real_label.append(tf.argmax(y_test, axis=1).numpy())
            test_prediction.append(_pred_y)
        num_test += 1

    for class_idx in range(0, param.num_class):
        _y = tf.one_hot(indices=tf.multiply(tf.ones(shape=(param.batch_size), dtype=tf.float32) * class_idx),
                        depth=param.num_class)
        _z = tf.random.uniform(shape=(param.batch_size, param.latent_dim), minval=-1.0, maxval=1.0,
                               dtype=tf.float32)

        _gen_input = tf.concat([_y, _z], axis=-1, name='gen_input')
        _x_tilde = generator(_gen_input, training=False)

        save_decode_image_array(_x_tilde.numpy(), path=os.path.join(graph_path,
                                                                    '{}_c{}_generated.png'.format(graph, class_idx)))

    # 7. Report
    train_loss_dis = np.mean(np.reshape(train_loss_dis, (-1)))
    train_loss_gen = np.mean(np.reshape(train_loss_gen, (-1)))
    train_loss_cla_real = np.mean(np.reshape(train_loss_cla_real, (-1)))
    train_loss_cla_fake = np.mean(np.reshape(train_loss_cla_fake, (-1)))
    train_was_x = np.mean(np.reshape(train_was_x, (-1)))

    val_loss_dis = np.mean(np.reshape(val_loss_dis, (-1)))
    val_loss_gen = np.mean(np.reshape(val_loss_gen, (-1)))
    val_loss_cla_real = np.mean(np.reshape(val_loss_cla_real, (-1)))
    val_loss_cla_fake = np.mean(np.reshape(val_loss_cla_fake, (-1)))
    val_was_x = np.mean(np.reshape(val_was_x, (-1)))

    test_loss_dis = np.mean(np.reshape(test_loss_dis, (-1)))
    test_loss_gen = np.mean(np.reshape(test_loss_gen, (-1)))
    test_loss_cla_real = np.mean(np.reshape(test_loss_cla_real, (-1)))
    test_loss_cla_fake = np.mean(np.reshape(test_loss_cla_fake, (-1)))
    test_was_x = np.mean(np.reshape(test_was_x, (-1)))

    train_real_label, train_prediction = np.reshape(train_real_label, (-1)), np.reshape(train_prediction, (-1))
    valid_real_label, valid_prediction = np.reshape(valid_real_label, (-1)), np.reshape(valid_prediction, (-1))
    test_real_label, test_prediction = np.reshape(test_real_label, (-1)), np.reshape(test_prediction, (-1))

    train_acc = metrics.accuracy_score(train_real_label, train_prediction)
    valid_acc = metrics.accuracy_score(valid_real_label, valid_prediction)
    test_acc = metrics.accuracy_score(test_real_label, test_prediction)

    print("[Loss cla fake] Train: {:.06f}\t Validation: {:.06f}\t Test: {:.06f}".format(train_loss_cla_fake,
                                                                                        val_loss_cla_fake,
                                                                                        test_loss_cla_fake))
    print("[Loss cla real] Train: {:.06f}\t Validation: {:.06f}\t Test: {:.06f}".format(train_loss_cla_real,
                                                                                        val_loss_cla_real,
                                                                                        test_loss_cla_real))
    print("[Loss dis] Train: {:.06f}\t Validation: {:.06f}\t Test: {:.06f}".format(train_loss_dis, val_loss_dis,
                                                                                   test_loss_dis))
    print("[Loss gen] Train: {:.06f}\t Validation: {:.06f}\t Test: {:.06f}".format(train_loss_gen, val_loss_gen,
                                                                                   test_loss_gen))

    print("[Was X] Train: {:.06f}\t Validation: {:.06f}\t Test: {:.06f}".format(train_was_x, val_was_x,
                                                                                test_was_x))
    print("[Accuracy] Train: {:.05f}\t Validation: {:.05f}\t Test: {:.05f}".format(train_acc, valid_acc, test_acc))

    # 8. Draw some samples
    save_decode_image_array(x_test.numpy(), path=os.path.join(graph_path, '{}_original.png'.format(graph)))
    save_decode_image_array(x_tilde.numpy(), path=os.path.join(graph_path, '{}_generated.png'.format(graph)))


if __name__ == '__main__':
    inference()
