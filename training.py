import matplotlib

matplotlib.use('Agg')

from copy import deepcopy
import os

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

from model_hvae_L import Model

from generation import generation_model

import gzip
import pickle


def training_model(model_name, train_data, valid_data, device, layers, x_dim, h_dim, z_dim, n_batch, eta, epochs,
                   n_pseudo_inputs, early_stopping, warm_up, weighted_vp_bool, He_bool, plot_show_bool=False):
    vae = Model(x_dim, h_dim, z_dim, n_batch, device, n_pseudo_inputs, layers, weighted_vp_bool, He_bool)
    vae.to(device)

    train_loss_lst, valid_loss_lst = [], []
    active_units_val_lst_per_layer_per_epoch = []
    kl_loss_lst_val, re_loss_lst_val = [], []

    parameters = list(vae.parameters())
    optimizer = optim.Adam(parameters, lr=eta, amsgrad=True)

    updates_per_epoch = int(train_data.shape[0] / float(n_batch))
    indicies = np.arange(train_data.shape[0])

    tqdm_epochs = tqdm(range(epochs))
    for epoch in tqdm_epochs:

        ### Warm-up (begin)
        if epoch == 0 or epoch >= warm_up:
            vae.beta = 1.0
        else:
            vae.beta = (epoch + 1.0) / warm_up
        ### Warm-up (end)

        ### Shuffling
        np.random.shuffle(indicies)

        for update in range(updates_per_epoch):
            x = Variable(
                torch.tensor(train_data[indicies[update * n_batch: (update + 1) * n_batch]]).float().to(device))

            optimizer.zero_grad()
            vae.forward(x)
            vae.compute_loss(x)
            vae.loss.backward()
            optimizer.step()

            if weighted_vp_bool:
                vae.w_pseudo_layer.weight = nn.Parameter(vae.softmax_layer(vae.w_pseudo_layer.weight))

            train_loss_lst.append(vae.loss.data)

        ### Early Stopping (begin)
        ###
        ### Early stopping is applied after finishing the warm-up
        ###

        z_dead_units_per_layer = np.zeros((layers, z_dim))
        temp_valid_loss, temp_kl_loss_val, temp_re_loss_val = 0, 0, 0
        for n in range(0, int(valid_data.shape[0] / n_batch)):
            x = Variable(torch.tensor(valid_data[n * n_batch: (n + 1) * n_batch]).float().to(device))
            vae.forward(x)
            vae.compute_loss(x)
            temp_valid_loss += vae.loss.data
            temp_kl_loss_val += vae.kl_loss.data
            temp_re_loss_val += vae.reconstruction_loss.data

            for en_layer, layer in enumerate(reversed(range(layers))):
                z_dead_units_per_layer[en_layer] += np.var(vae.en_q_z_mean_layer[layer].cpu().data.numpy(), axis=0)

        z_dead_units_per_layer /= (valid_data.shape[0] / n_batch)

        active_units_per_layer = []
        for layer in range(layers):
            active_units_per_layer.append(sum(z_dead_units_per_layer[layer] > 0.01))
        active_units_val_lst_per_layer_per_epoch.append(active_units_per_layer)

        valid_loss_lst.append(temp_valid_loss * n_batch / valid_data.shape[0])
        kl_loss_lst_val.append(temp_kl_loss_val * n_batch / valid_data.shape[0])
        re_loss_lst_val.append(temp_re_loss_val * n_batch / valid_data.shape[0])

        if early_stopping is not None and epoch > warm_up:
            if len(valid_loss_lst[warm_up:]) - np.argmin(valid_loss_lst[warm_up:]) > early_stopping:
                print("Warm up is used for {} epochs".format(warm_up))
                print("Breaking because of early stopping after {} epochs".format(epoch))
                print("Best model achieved at {} epoch".format(epoch - early_stopping))
                break

            if len(valid_loss_lst[warm_up:]) - 1 == np.argmin(valid_loss_lst[warm_up:]):
                best_model_dict = vae.state_dict()

        elif len(valid_loss_lst) - 1 == np.argmin(valid_loss_lst):
            best_model_dict = vae.state_dict()

        ### Early Stopping (end)

        if weighted_vp_bool:
            tqdm_epochs.set_description(
                "Epoch:{} Training Loss:{} Validation Loss:{}, W:{}".format(
                    epoch, train_loss_lst[-1], valid_loss_lst[-1], vae.w_pseudo_layer.weight))
        else:
            tqdm_epochs.set_description(
                "Epoch:{} Training Loss:{} Validation Loss:{}".format(
                    epoch, train_loss_lst[-1], valid_loss_lst[-1]))

        if epoch % 50 != 0:
            continue

        fig = plt.figure("Generated Digits")
        gs = gridspec.GridSpec(4, 4)

        samples = [deepcopy(sample.to("cpu").data.numpy()) for sample in vae.p_x_mean[:16]]

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28, 28), cmap='gray')
        plt.tight_layout()
        plt.savefig(os.path.join("Results", model_name, "plot_digits_{}.png".format(epoch)))
        if plot_show_bool:
            plt.pause(1e-6)
        plt.close(fig)

        generation_model(vae, model_name, n_digits=16, epoch=epoch)

        if epoch == 0:
            continue

        fig = plt.figure("loss")
        plt.clf()
        plt.title("Loss for epoch:{}".format(epoch))
        plt.plot(range(1, epoch + 2), train_loss_lst[updates_per_epoch - 1::updates_per_epoch])
        plt.plot(range(1, epoch + 2), valid_loss_lst)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(["training loss", "validation loss"])
        plt.savefig(os.path.join("Results", model_name, "loss_{}.png".format(epoch)))
        if plot_show_bool:
            plt.pause(1e-6)
        plt.close(fig)

    vae.load_state_dict(best_model_dict)
    torch.save(best_model_dict, os.path.join("Results", model_name, "best_model.pickle"))

    with gzip.open(os.path.join("Results", model_name, "training_loss_per_update.pickle.gzip"), "wb") as fl:
        pickle.dump(train_loss_lst, fl)

    with gzip.open(os.path.join("Results", model_name, "validation_loss_per_epoch.pickle.gzip"), "wb") as fl:
        pickle.dump(valid_loss_lst, fl)

    with gzip.open(os.path.join("Results", model_name, "active_units_val_lst_per_epoch_per_layer.pickle.gzip"),
                   "wb") as fl:
        pickle.dump(active_units_val_lst_per_layer_per_epoch, fl)

    with gzip.open(os.path.join("Results", model_name, "kl_loss_per_epoch_val.pickle.gzip"), "wb") as fl:
        pickle.dump(kl_loss_lst_val, fl)

    with gzip.open(os.path.join("Results", model_name, "re_loss_per_epoch_val.pickle.gzip"), "wb") as fl:
        pickle.dump(re_loss_lst_val, fl)

    return vae, train_loss_lst, valid_loss_lst
