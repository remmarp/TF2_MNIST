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
    max_epoch = 10000
    valid_step = 20
    batch_size = 128
    num_early_stopping = 20

    input_dim = (28, 28, 1)
    latent_dim = 256

    learning_rate_class = 3e-4

    model_display_len = 150

    num_class = 10
