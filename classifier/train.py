# -*- coding: utf-8 -*-
# """
# classifier/train.py
# """

############
#   IMPORT #
############
# 1. Built-in modules
import os
import time
import sys

sys.path.append(os.getcwd())

# 2. Third-party modules
import numpy as np
import tensorflow as tf

# 3. Own modules
from data_loader import MNISTLoader
from classifier.parameter import Parameter
from classifier.networks import Classifier

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


################
#   DEFINITION #
################
def train():
    param = Parameter()

    # 1. Build models
    classifier = Classifier(param).model()

    # 2. Set optimizers
    opt_class = tf.keras.optimizers.Adam(learning_rate=param.learning_rate_class)

    # 3. Set trainable variables
    var_class = classifier.trainable_variables

    # 4. Load data
    data_loader = MNISTLoader(one_hot=False)
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

    minimum_loss = 100.
    num_effective_epoch = 0

    net_name = 'classifier'
    check_point_prefix = os.path.join(check_point_dir, 'classifier')
    check_point = tf.train.Checkpoint(opt_class=opt_class, classifier=classifier)
    ckpt_manager = tf.train.CheckpointManager(check_point, check_point_dir, max_to_keep=5,
                                              checkpoint_name=check_point_prefix)

    # 7. Define train / validation step ################################################################################
    def training_step(_x, _y):
        with tf.GradientTape() as _class_tape:
            _prediction = classifier(_x, training=True)
            _loss_class = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=_y, logits=_prediction))

        _grad_class = _class_tape.gradient(_loss_class, var_class)
        opt_class.apply_gradients(zip(_grad_class, var_class))

    def validation_step(_x, _y):
        _prediction = classifier(_x, training=False)
        _loss_class = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=_y, logits=_prediction))

        return _prediction, _loss_class.numpy()
    ####################################################################################################################

    # 8. Train
    start_time = time.time()
    for epoch in range(0, param.max_epoch):
        # 8-1. Train classifier
        for x_train, y_train in train_set:
            training_step(x_train, y_train)

        # 8-2. Validation
        num_valid = 0
        softmax_cc_valid = []
        for x_valid, y_valid in test_set:
            prediction, loss_class = validation_step(x_valid, y_valid)
            softmax_cc_valid.append(loss_class)

            if num_valid == param.valid_step:
                break
            num_valid += 1

        valid_loss = np.mean(softmax_cc_valid)

        save_message = ''
        if minimum_loss > valid_loss:
            num_effective_epoch = 0
            minimum_loss = valid_loss
            save_message = "\tSave model: detecting lowest cross entropy: {:.6f} at epoch {:04d}".format(minimum_loss,
                                                                                                         epoch)

            classifier.save_weights(os.path.join(model_path, net_name))

        elapsed_time = (time.time() - start_time) / 60.

        # 8-3. Report
        print("[Epoch: {:04d}] {:.01f} min. class loss: {:.6f} Effective: {}".format(epoch, elapsed_time,
                                                                                     valid_loss,
                                                                                     (num_effective_epoch == 0)))
        print("{}".format(save_message))

        if epoch % param.save_frequency == 0 and epoch > 1:
            ckpt_manager.save(checkpoint_number=epoch)

        if num_effective_epoch >= param.num_early_stopping:
            print("\t Early stopping at epoch {:04d}!".format(epoch))
            break

        num_effective_epoch += 1


if __name__ == '__main__':
    train()
