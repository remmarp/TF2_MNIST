# -*- coding: utf-8 -*-
# """
# adversarial_autoencoder/parameter.py
# """

############
#   IMPORT #
############
# 1. Built-in modules
import os
# 2. Third-party modules
# 3. Own modules


###########
#   CLASS #
###########
class Parameter(object):
    max_epoch = 200
    pre_train_epoch = 5
    valid_step = 20
    batch_size = 256

    input_dim = (28, 28, 1)
    latent_dim = 100
    num_class = 10

    learning_rate_cla = 5e-5
    learning_rate_gen = 5e-5
    learning_rate_dis = 1e-5

    model_display_len = 150

    w_gp_lambda = 0.1
    prior_noise_std = 0.3

    cur_dir = os.getcwd()
    save_frequency = 10
