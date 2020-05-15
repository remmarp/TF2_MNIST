# -*- coding: utf-8 -*-
# """
# generative_adversarial_networks/parameter.py
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
    valid_step = 20
    batch_size = 128

    input_dim = (28, 28, 1)
    latent_dim = 256

    learning_rate_gen = 1e-5
    learning_rate_dis = 1e-5

    model_display_len = 150

    w_gp_lambda = 0.1

    cur_dir = os.path.join(os.getcwd(), 'generative_adversarial_networks')
