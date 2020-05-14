# -*- coding: utf-8 -*-
# """
# classifier/inference.py
# """

############
#   IMPORT #
############
# 1. Built-in modules
import os

# 2. Third-party modules
import numpy as np
import tensorflow as tf
import sklearn.metrics as metrics

# 3. Own modules
from data_loader import MNISTLoader
from classifier.parameter import Parameter
from classifier.networks import Classifier

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


################
#   DEFINITION #
################
def inference():
    param = Parameter()

    # 1. Build models
    classifier = Classifier(param).model()

    classifier.summary(line_length=param.model_display_len)

    # 2. Load data
    data_loader = MNISTLoader(one_hot=False)
    train_set = data_loader.train.batch(batch_size=param.batch_size, drop_remainder=True)
    test_set = data_loader.test.batch(batch_size=param.batch_size, drop_remainder=True)

    # 3. Etc.
    graph_path = os.path.join(os.getcwd(), 'graph')
    if not os.path.isdir(graph_path):
        os.mkdir(graph_path)

    model_path = os.path.join(os.getcwd(), 'model')
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    # 4. Load model
    _net = 'cc_sparse_softmax_cross_entropy_classifier'
    # graph = ''

    classifier.load_weights(os.path.join(model_path, _net))

    # 5. Define loss

    # 6. Inference
    train_loss = []

    train_real_label, train_prediction = [], []
    for x_train, y_train in train_set:
        prediction = classifier(x_train, training=False)

        loss_class = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_train,
                                                                                   logits=prediction))
        train_loss.append(loss_class.numpy())

        _pred_y = tf.argmax(prediction, axis=1)

        train_real_label.append(y_train.numpy())
        train_prediction.append(_pred_y.numpy())

    num_test = 0
    valid_loss = []
    test_loss = []

    valid_real_label, valid_prediction = [], []
    test_real_label, test_prediction = [], []

    for x_test, y_test in test_set:
        prediction = classifier(x_test, training=False)

        loss_class = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_test,
                                                                                   logits=prediction))
        _pred_y = tf.argmax(prediction, axis=1)
        if num_test <= param.valid_step:
            valid_loss.append(loss_class.numpy())
            valid_prediction.append(_pred_y.numpy())
            valid_real_label.append(y_test.numpy())

        else:
            test_loss.append(loss_class.numpy())
            test_prediction.append(_pred_y.numpy())
            test_real_label.append(y_test.numpy())
        num_test += 1

    # 7. Report
    train_real_label, train_prediction = np.reshape(train_real_label, (-1)), np.reshape(train_prediction, (-1))
    valid_real_label, valid_prediction = np.reshape(valid_real_label, (-1)), np.reshape(valid_prediction, (-1))
    test_real_label, test_prediction = np.reshape(test_real_label, (-1)), np.reshape(test_prediction, (-1))

    train_acc = metrics.accuracy_score(train_real_label, train_prediction)
    valid_acc = metrics.accuracy_score(valid_real_label, valid_prediction)
    test_acc = metrics.accuracy_score(test_real_label, test_prediction)

    print("[Loss] Train: {:.06f}\t Validation: {:.06f}\t Test: {:.06f}".format(np.mean(train_loss), np.mean(valid_loss),
                                                                               np.mean(test_loss)))
    print("[Accuracy] Train: {:.05f}\t Validation: {:.05f}\t Test: {:.05f}".format(train_acc, valid_acc, test_acc))

    # 8. Draw some samples


if __name__ == '__main__':
    inference()
