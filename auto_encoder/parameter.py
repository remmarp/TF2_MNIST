# -*- coding: utf-8 -*-
# """
# auto_encoder/param.py
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
    max_epoch = 10000
    valid_step = 20
    batch_size = 128
    num_early_stopping = 20

    input_dim = (28, 28, 1)
    latent_dim = 256

    white_noise_std = 0.2

    learning_rate_ae = 2e-3

    model_display_len = 150

    cur_dir = os.getcwd()
    save_frequency = 10
