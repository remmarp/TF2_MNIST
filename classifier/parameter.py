# -*- coding: utf-8 -*-
# """
# classifier/parameter.py
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
    valid_step = 30
    batch_size = 128
    num_early_stopping = 20

    input_dim = (28, 28, 1)

    learning_rate_class = 3e-4

    model_display_len = 150

    num_class = 10

    cur_dir = os.path.join(os.getcwd(), 'classifier')
