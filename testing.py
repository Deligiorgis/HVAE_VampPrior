import matplotlib

matplotlib.use('Agg')

import os

import numpy as np

import matplotlib.pyplot as plt

import gzip
import pickle



def measure_active_units_model(X, model):
    model.measure_dead_units(X)
    active_units_per_layer = []
    for layer in range(model.layers):
        active_units_per_layer.append(sum(model.z_dead_units_per_layer[layer] > 0.01))
        print("Active units:{} for layer:{}".format(active_units_per_layer[layer], layer + 1))

    return active_units_per_layer


def measure_marginal_LL_model(X, model, model_name, data_name, batch_size=500):
    print("Calculating the marginal log-likelihodd")
    model.marginal_LL(X, batch_size=batch_size)
    average_marginal_LL = np.mean(model.marg_ll)
    print("The averaged marginal log-likelihood is {}\n".format(average_marginal_LL))

    plt.figure("Histogram")
    plt.clf()
    plt.hist(model.marg_ll, 100, density=True, facecolor="blue", alpha=0.5)
    plt.ylabel("Probability")
    plt.xlabel("Marginal Log-Likelihood")
    plt.title("Marginal Log-Likelihood for the {} dataset".format(data_name))
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join("Results", model_name, "{}_histogram.png".format(data_name)))

    with gzip.open(os.path.join("Results", model_name, "{}_data_for_histogram.pickle.gzip".format(data_name)),
                   "wb") as fl:
        pickle.dump(model.marg_ll, fl)

    return average_marginal_LL
