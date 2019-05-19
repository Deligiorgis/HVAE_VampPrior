import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np

from copy import deepcopy

import os


def generation_model(model, model_name, n_digits=16, epoch=None):
    model.generative(n_digits=n_digits)

    fig = plt.figure("Generated Digits from generative")
    gs = gridspec.GridSpec(int(np.sqrt(n_digits)), int(np.sqrt(n_digits)))

    samples = [deepcopy(sample.to("cpu").data.numpy()) for sample in model.p_x_mean]

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='gray')
    plt.tight_layout()
    if epoch is None:
        plt.savefig(os.path.join("Results", model_name, "plot_digits_generative.png"))
    else:
        plt.savefig(os.path.join("Results", model_name, "plot_digits_generative_epoch_{}.png".format(epoch)))
    plt.close(fig)

    return samples
