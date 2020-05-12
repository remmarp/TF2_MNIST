# -*- coding: utf-8 -*-
# """
# param.py
# """

############
#   IMPORT #
############
# 1. Built-in modules
# 2. Third-party modules
# 3. Own modules


###########
#   CLASS #
###########
class Parameter(object):
    max_epoch = 10000
    valid_step = 20
    batch_size = 128
    num_early_stopping = 20

    input_dim = (28, 28, 1)
    latent_dim = 256

    white_noise_std = 0.2

    learning_rate_ae = 3e-4

    train_freq_ae = 5

    model_display_len = 150
