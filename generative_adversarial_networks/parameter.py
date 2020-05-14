# -*- coding: utf-8 -*-
# """
# classifier/parameter.py
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
    max_epoch = 50
    valid_step = 20
    batch_size = 128

    input_dim = (28, 28, 1)
    latent_dim = 100

    learning_rate_gen = 1e-4
    learning_rate_dis = 1e-4

    model_display_len = 150

    w_gp_lambda = 0.1
