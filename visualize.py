# -*- coding: utf-8 -*-
# """
# visualize.py
# """

############
#   IMPORT #
############
# 1. Built-in modules

# 2. Third-party modules
import numpy as np
import matplotlib.pyplot as plt

# 3. Own modules


##############
#   DEFINITION
##############
# Save decoded MNIST image
def save_decode_image_array(d, path, cols=4):
    # image_size = int(np.sqrt(len(d[0])))
    w, h = d.shape[1], d.shape[2]

    rows = cols
    plt.clf()
    plt.gray()

    fig, axes = plt.subplots(ncols=cols, nrows=rows)
    for r in range(rows):
        for c in range(cols):
            axes[r, c].imshow(d[r * cols + c, :].reshape(w, h) * 127.5 + 127.5, cmap='gray')
            axes[r, c].set(adjustable='box', aspect='equal')
            axes[r, c].get_xaxis().set_visible(False)
            axes[r, c].get_yaxis().set_visible(False)

    plt.savefig(path)
    plt.close()
