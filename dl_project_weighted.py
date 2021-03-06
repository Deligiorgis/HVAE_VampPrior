# -*- coding: utf-8 -*-
"""DL_Project.ipynb

Automatically generated by Colaboratory.

# Imports
"""

import gzip
import pickle
import os
import urllib.request

from copy import deepcopy

import numpy as np

from scipy.special import logsumexp

from tqdm.auto import tqdm

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from datetime import datetime

if not os.path.exists("Results"):
    os.makedirs("Results")

"""# Classes & Functions"""

def read_pickled_data_gzip():
    path_pickled_data = os.path.join("..", "Data", "Pickle_binarized_mnist")
    filenames = tqdm(os.listdir(path_pickled_data))
    dict_names = {}
    for filename in filenames:
        if "gzip" not in filename:
            continue
        dataset = filename.split("_")[-1].split(".")[0]
        filenames.set_description("Reading compressed pickled: {}".format(dataset))
        with gzip.open(os.path.join(path_pickled_data, filename), "rb") as fl:
            dict_names[dataset] = pickle.load(fl)
    train_data, valid_data, test_data = [dict_names[key] for key in ["train", "valid", "test"]]
    return train_data, valid_data, test_data

"""## Read Data

## GLU & Layer
"""

class Layer_He(nn.Module):

    def __init__(self, input_dim, output_dim, activation=None, initialization=nn.init.kaiming_normal_,
                 mu=None, std=None, bias = True):
        super(Layer_He, self).__init__()

        self.activation = activation
        self.layer_model = nn.Linear(input_dim, output_dim, bias = bias)

        if initialization == nn.init.kaiming_normal_:
            initialization(self.layer_model.weight)
        elif initialization == nn.init.normal_:
            mu = 0.0 if mu is None else mu
            std = 1.0 if std is None else std
            initialization(self.layer_model.weight, mu, std)
            
    def forward(self, x):
        self.layer = self.layer_model(x)
        if self.activation is not None:
            return self.activation(self.layer)
        return self.layer


class GLU_He(nn.Module):

    def __init__(self, input_dim, output_dim, gate_activation=torch.sigmoid, activation=None,
                 initialization=nn.init.kaiming_normal_):
        super(GLU_He, self).__init__()

        self.gate_activation = gate_activation
        self.activation = activation
        self.L_model = nn.Linear(input_dim, output_dim)
        self.G_model = nn.Linear(input_dim, output_dim)

        initialization(self.L_model.weight)
        initialization(self.G_model.weight)

    def forward(self, x):
        self.L = self.L_model(x)

        self.G = self.gate_activation(self.G_model(x))

        if self.activation is not None:
            self.L = self.activation(self.L)

        return self.L * self.G

      

class Layer_Xa(nn.Module):

    def __init__(self, input_dim, output_dim, activation=None, initialization=nn.init.xavier_normal_,
                 mu=None, std=None, bias = True):
        super(Layer_Xa, self).__init__()

        self.activation = activation
        self.layer_model = nn.Linear(input_dim, output_dim, bias = bias)

        if initialization == nn.init.xavier_normal_:
            initialization(self.layer_model.weight)
        elif initialization == nn.init.normal_:
            mu = 0.0 if mu is None else mu
            std = 1.0 if std is None else std
            initialization(self.layer_model.weight, mu, std)

    def forward(self, x):
        self.layer = self.layer_model(x)

        if self.activation is not None:
            return self.activation(self.layer)
        return self.layer


class GLU_Xa(nn.Module):

    def __init__(self, input_dim, output_dim, gate_activation=torch.sigmoid, activation=None,
                 initialization=nn.init.xavier_normal_):
        super(GLU_Xa, self).__init__()

        self.gate_activation = gate_activation
        self.activation = activation
        self.L_model = nn.Linear(input_dim, output_dim)
        self.G_model = nn.Linear(input_dim, output_dim)

        initialization(self.L_model.weight)
        initialization(self.G_model.weight)

    def forward(self, x):
        self.L = self.L_model(x)

        self.G = self.gate_activation(self.G_model(x))

        if self.activation is not None:
            self.L = self.activation(self.L)

        return self.L * self.G

class Layer_weight(nn.Module):
  
    def __init__(self, input_dim, output_dim,device):
        super(Layer_weight, self).__init__()
        
        self.linear_layer = nn.Linear(input_dim, output_dim, bias=False)
        self.device = device     
          
        nn.init.kaiming_normal_(self.linear_layer.weight)
        self.normalize = lambda x : (x + torch.min(x)*2)/ torch.sum(x+torch.min(x)*2)
        self.linear_layer.weight = nn.Parameter(self.normalize(self.linear_layer.weight))
#        print(torch.sum(self.linear_layer.weight))
#        print(self.linear_layer.weight)
        
        self.norm = lambda x : x / torch.sum(x)
        
    def log_normal(self, x, mu, log_var):
        return - 0.5 * (log_var + (x - mu) ** 2 / torch.exp(log_var)) 
        
    def forward(self, z_u, z_given_u_mean, z_given_u_log_var):
        self.log_N_z = self.log_normal(z_u, z_given_u_mean, z_given_u_log_var)  # (100, 500, 40)
        self.log_N_z = torch.sum(self.log_N_z, 2)  # (100, 500)
        self.log_N_z_max, _ = torch.max(self.log_N_z, 1)
        #self.linear_layer.weight = nn.Parameter(self.norm(self.linear_layer.weight))
        self.calc = self.linear_layer(torch.exp(self.log_N_z - self.log_N_z_max.unsqueeze(1)))
        #min = torch.min(torch.where(self.calc<0,torch.tensor(1e10,device=self.device),self.calc))

        self.N_z = torch.where(self.calc>0,self.calc,torch.tensor(1e-100,device=self.device))
        return self.log_N_z_max + torch.log(self.N_z)

"""## HVAE (L=K)"""

class Model(nn.Module):

    def __init__(self, x_dim, h_dim, z_dim, n_batch, device, n_pseudo_inputs, layers, weighted_vp_bool, He_bool, Lambda=0):
        super(Model, self).__init__()

        if He_bool:
            print("He initialization \t is used")
            GLU, Layer = GLU_He, Layer_He
        else:
            print("Xavier initialization \t is used")
            GLU, Layer = GLU_Xa, Layer_Xa

        self.layers = layers
        self.n_batch = n_batch
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.device = device
        self.beta = 1.0
        self.weighted_vp_bool = weighted_vp_bool
        
        self.Lambda = Lambda

        self.vampprior_bool = False
        if n_pseudo_inputs > 0:
            print("VampPrior \t is used as a prior")
            self.vampprior_bool = True
            self.n_pseudo_inputs = n_pseudo_inputs
            self.input_mean = Variable(torch.eye(self.n_pseudo_inputs, device=self.device), requires_grad=False)
            self.pseudo_means = Layer(self.n_pseudo_inputs, self.x_dim, activation=nn.Hardtanh(min_val=0, max_val=1),
                                      initialization=nn.init.normal_, std=0.01, bias = False)

            if self.weighted_vp_bool:
                print("Weighted VampPrior \t is used")
                self.w_pseudo_layer = Layer_weight(self.n_pseudo_inputs, 1, self.device)
         #       nn.init.kaiming_normal_(self.w_pseudo_layer.weight)
                #nn.init.normal(self.w_pseudo_layer.weight, mean=0, std=10) 
                #print('Init: ',self.w_pseudo_layer.weight)

                #self.softmax_layer = nn.Softmax(dim=1)
                #self.softmax_layer = lambda x : (x + torch.min(x)*2)/ torch.sum(x+torch.min(x)*2)
                
                #self.normalize =lambda x : torch.abs(x)
           #     self.normalize = lambda x : (x + torch.min(x)*2)/ torch.sum(x+torch.min(x)*2)
                
                #self.norm = lambda x : x / torch.sum(x)
                #self.softmax_layer = lambda x : self.norm(torch.clamp(x, min=1e-5, max=1 - 1e-5))
            #    self.w_pseudo_layer.weight = nn.Parameter(self.normalize(self.w_pseudo_layer.weight))
               # print('Init+softm: ',self.w_pseudo_layer.weight)


        else:
            print("Standard Gaussian \t is used as a prior")

        ### Encoder: q ( z_L | x) Π q ( z_i | x, z_{i+1}, ..., z_L ) i = 1, .., L - 1 (Variational)
        self.q_z_given_x_hidden_layer = nn.ModuleList()
        self.q_z_given_z_plus_hidden_per_layer = nn.ModuleList()
        self.q_z_given_x_and_z_hidden_layer = nn.ModuleList()
        self.q_z_mean_layer = nn.ModuleList()
        self.q_z_log_var_layer = nn.ModuleList()

        for layer in range(self.layers):
            self.q_z_given_x_hidden_layer.append(GLU(self.x_dim, self.h_dim).to(self.device))
            q_z_given_z_plus_hidden_layer = nn.ModuleList()
            for sub_layer in range(layer + 1, self.layers):
                q_z_given_z_plus_hidden_layer.append(GLU(self.z_dim, self.h_dim).to(self.device))
            self.q_z_given_z_plus_hidden_per_layer.append(q_z_given_z_plus_hidden_layer)
            self.q_z_given_x_and_z_hidden_layer.append(
                GLU((self.layers - layer) * self.h_dim, self.h_dim).to(self.device))

            # self.q_z_mean_layer.append(GLU(self.h_dim, self.z_dim).to(self.device))
            self.q_z_mean_layer.append(Layer(self.h_dim, self.z_dim).to(self.device))

            # self.q_z_log_var_layer.append(
            #     GLU(self.h_dim, self.z_dim, activation=nn.Hardtanh(min_val=-7, max_val=2.3)).to(self.device))
            self.q_z_log_var_layer.append(
                Layer(self.h_dim, self.z_dim, activation=nn.Hardtanh(min_val=-7, max_val=2.3)).to(self.device))

        ### Decoder p ( x | z ) p ( z_L ) Π p ( z_i | z_{i+1}, ..., z_L ) i = 1, .., L - 1 (Generative)
        self.p_z_given_z_plus_hidden_per_layer = nn.ModuleList()
        self.p_z_given_all_z_plus_hidden_layer = nn.ModuleList()
        self.p_z_mean_layer = nn.ModuleList()
        self.p_z_log_var_layer = nn.ModuleList()

        self.p_x_given_z_hidden_layer = nn.ModuleList()
        self.p_x_given_z_hidden_layer.append(GLU(self.z_dim, self.h_dim).to(self.device))

        for layer in range(self.layers - 1):
            p_z_given_z_plus_hidden_layer = nn.ModuleList()
            for sub_layer in range(layer + 1, self.layers):
                p_z_given_z_plus_hidden_layer.append(GLU(self.z_dim, self.h_dim)).to(self.device)
            self.p_z_given_z_plus_hidden_per_layer.append(p_z_given_z_plus_hidden_layer)
            self.p_z_given_all_z_plus_hidden_layer.append(
                GLU((self.layers - 1 - layer) * self.h_dim, self.h_dim).to(self.device))

            # self.p_z_mean_layer.append(GLU(self.h_dim, self.z_dim).to(self.device))
            self.p_z_mean_layer.append(Layer(self.h_dim, self.z_dim).to(self.device))

            # self.p_z_log_var_layer.append(
            #     GLU(self.h_dim, self.z_dim, activation=nn.Hardtanh(min_val=-7, max_val=2.3)).to(self.device))
            self.p_z_log_var_layer.append(
                Layer(self.h_dim, self.z_dim, activation=nn.Hardtanh(min_val=-7, max_val=2.3)).to(self.device))

            self.p_x_given_z_hidden_layer.append(GLU(self.z_dim, self.h_dim).to(self.device))

        self.p_x_given_all_z_hidden_layer = GLU(self.layers * self.h_dim, self.h_dim).to(self.device)
        self.p_x_mean_mlp = Layer(self.h_dim, self.x_dim, activation=torch.sigmoid).to(self.device)

    def reparameterization(self, z_mean, z_log_var):
        self.epsilon = Variable(torch.randn((self.n_batch, self.z_dim), device=self.device))
        self.z = z_mean + torch.exp(z_log_var / 2.0) * self.epsilon
        return self.z

    def encoder(self, x):

        ### They are reversed
        self.en_q_z_given_x_hidden_layer = []
        self.en_q_z_given_z_plus_hidden_per_layer = []
        self.en_q_z_cat_hidden_layer = []
        self.en_q_z_given_x_and_z_hidden_layer = []
        self.en_q_z_mean_layer = []
        self.en_q_z_log_var_layer = []
        self.q_z_layer = []  ### After reparameterization

        for en_layer, layer in enumerate(reversed(range(self.layers))):
            self.en_q_z_given_x_hidden_layer.append(self.q_z_given_x_hidden_layer[layer](x))
            en_q_z_given_z_plus_hidden_per_layer = []
            for sub_layer in range(layer + 1, self.layers):
                en_q_z_given_z_plus_hidden_per_layer.append(
                    self.q_z_given_z_plus_hidden_per_layer[layer][self.layers - 1 - sub_layer](
                        self.q_z_layer[self.layers - 1 - sub_layer]))
            self.en_q_z_given_z_plus_hidden_per_layer.append(en_q_z_given_z_plus_hidden_per_layer)
            self.en_q_z_cat_hidden_layer.append(
                torch.cat(
                    (self.en_q_z_given_x_hidden_layer[en_layer], *self.en_q_z_given_z_plus_hidden_per_layer[en_layer]),
                    dim=1))
            self.en_q_z_given_x_and_z_hidden_layer.append(
                self.q_z_given_x_and_z_hidden_layer[layer](self.en_q_z_cat_hidden_layer[en_layer]))
            self.en_q_z_mean_layer.append(self.q_z_mean_layer[layer](self.en_q_z_given_x_and_z_hidden_layer[en_layer]))
            self.en_q_z_log_var_layer.append(
                self.q_z_log_var_layer[layer](self.en_q_z_given_x_and_z_hidden_layer[en_layer]))
            self.q_z_layer.append(
                self.reparameterization(self.en_q_z_mean_layer[en_layer], self.en_q_z_log_var_layer[en_layer]))

    def decoder(self):

        ### They are reversed
        self.de_p_z_given_z_plus_hidden_per_layer = []
        self.de_p_z_cat_hidden_layer = []
        self.de_p_z_given_all_z_plus_hidden_layer = []
        self.de_p_z_mean_layer = []
        self.de_p_z_log_var_layer = []
        self.p_z_layer = []

        self.de_p_x_given_z_hidden_layer = []
        self.de_p_x_given_z_hidden_layer.append(
            self.p_x_given_z_hidden_layer[-1](self.q_z_layer[0]))  # 0 == z_L because it is reversed

        for en_layer, layer in enumerate(reversed(range(self.layers - 1))):
            de_p_z_given_z_plus_hidden_layer = []
            for sub_layer in range(layer + 1, self.layers):
                de_p_z_given_z_plus_hidden_layer.append(
                    self.p_z_given_z_plus_hidden_per_layer[layer][self.layers - sub_layer - 1](
                        self.q_z_layer[0] if sub_layer == self.layers - 1 else self.p_z_layer[
                            self.layers - sub_layer - 2]))
            self.de_p_z_given_z_plus_hidden_per_layer.append(de_p_z_given_z_plus_hidden_layer)
            self.de_p_z_cat_hidden_layer.append(
                torch.cat((*self.de_p_z_given_z_plus_hidden_per_layer[en_layer], torch.tensor([]).to(self.device)),
                          dim=1))
            self.de_p_z_given_all_z_plus_hidden_layer.append(
                self.p_z_given_all_z_plus_hidden_layer[layer](self.de_p_z_cat_hidden_layer[en_layer]))
            self.de_p_z_mean_layer.append(
                self.p_z_mean_layer[layer](self.de_p_z_given_all_z_plus_hidden_layer[en_layer]))
            self.de_p_z_log_var_layer.append(
                self.p_z_log_var_layer[layer](self.de_p_z_given_all_z_plus_hidden_layer[en_layer]))
            self.p_z_layer.append(
                self.reparameterization(self.de_p_z_mean_layer[en_layer], self.de_p_z_log_var_layer[en_layer]))

            self.de_p_x_given_z_hidden_layer.append(self.p_x_given_z_hidden_layer[layer](self.p_z_layer[en_layer]))

        self.de_p_x_cat_hidden = torch.cat((*self.de_p_x_given_z_hidden_layer, torch.tensor([]).to(self.device)), dim=1)

        self.de_p_x_given_all_z_hidden_layer = self.p_x_given_all_z_hidden_layer(self.de_p_x_cat_hidden)

        self.p_x_mean = self.p_x_mean_mlp(self.de_p_x_given_all_z_hidden_layer)

    def forward(self, x):
        self.encoder(x)
        self.decoder()
        return self.p_x_mean

    def log_normal(self, x, mu, log_var):
        return - 0.5 * (log_var + (x - mu) ** 2 / torch.exp(log_var))

    def log_vampprior(self):
        self.u = self.pseudo_means(self.input_mean)
        self.q_z_given_u_hidden_1st = self.q_z_given_x_hidden_layer[-1](self.u)
        self.q_z_given_u_hidden_2nd = self.q_z_given_x_and_z_hidden_layer[-1](self.q_z_given_u_hidden_1st)

        self.z_given_u_mean = self.q_z_mean_layer[-1](self.q_z_given_u_hidden_2nd)
        self.z_given_u_log_var = self.q_z_log_var_layer[-1](self.q_z_given_u_hidden_2nd)

        self.z_given_u_mean = self.z_given_u_mean.unsqueeze(0)
        self.z_given_u_log_var = self.z_given_u_log_var.unsqueeze(0)

        self.z_u = self.q_z_layer[0].unsqueeze(1)

        if self.weighted_vp_bool:
            '''
            self.log_N_z = self.log_normal(self.z_u, self.z_given_u_mean, self.z_given_u_log_var)  # (100, 500, 40)
            self.log_N_z = torch.sum(self.log_N_z, 2)  # (100, 500)
            self.log_N_z_max, _ = torch.max(self.log_N_z, 1)
            
            self.w_pseudo_layer.weight = nn.Parameter(torch.clamp(self.w_pseudo_layer.weight, min=1e-6))
            self.N_z = self.w_pseudo_layer(torch.exp(self.log_N_z - self.log_N_z_max.unsqueeze(1)))
            self.log_prior = self.log_N_z_max + torch.log(self.N_z)
            '''
            self.log_prior = self.w_pseudo_layer(self.z_u, self.z_given_u_mean, self.z_given_u_log_var)            
        else:
            a = torch.sum(self.log_normal(self.z_u, self.z_given_u_mean, self.z_given_u_log_var), 2) - np.log(
                self.n_pseudo_inputs)
            self.log_prior = torch.logsumexp(a, dim=1)

        return self.log_prior

    def kl_divergence(self, mu_1, log_var_1, mu_2, log_var_2):
        var_1, var_2 = torch.exp(log_var_1), torch.exp(log_var_2)
        kl_loss = torch.mean(
            0.5 * (torch.sum(log_var_2 - log_var_1 + var_1 / var_2 - 1 + (mu_2 - mu_1) ** 2 / var_2, 1)))
        return kl_loss

    def compute_loss(self, x):
        p_x_mean = torch.clamp(self.p_x_mean, min=1e-5, max=1 - 1e-5).double()
        self.reconstruction_loss = - torch.sum(
            x.double() * torch.log(p_x_mean) + (1. - x.double()) * torch.log(1. - p_x_mean)) / self.n_batch

        ### The kl_loss_layer is reversed
        self.kl_loss_layer = []
        for layer in range(1, self.layers):
            kl_loss_layer = self.kl_divergence(self.en_q_z_mean_layer[layer], self.en_q_z_log_var_layer[layer],
                                               self.de_p_z_mean_layer[layer - 1], self.de_p_z_log_var_layer[layer - 1])
            self.kl_loss_layer.append(kl_loss_layer)

        if self.vampprior_bool:
            self.kl_loss_prior = torch.mean(
                torch.sum(self.log_normal(self.q_z_layer[0], self.en_q_z_mean_layer[0], self.en_q_z_log_var_layer[0]),
                          1) - self.log_vampprior())
        else:
            self.kl_loss_prior = self.kl_divergence(self.en_q_z_mean_layer[0], self.en_q_z_log_var_layer[0],
                                                    torch.zeros_like(self.en_q_z_mean_layer[0]),
                                                    torch.zeros_like(self.en_q_z_log_var_layer[0]))

        self.kl_loss = self.kl_loss_prior + sum(self.kl_loss_layer)
        self.loss = self.reconstruction_loss.float() + self.beta * self.kl_loss + self.Lambda * (torch.sum(torch.abs(self.w_pseudo_layer.linear_layer.weight)) - 1)
        return self.loss

    def generative(self, n_digits):

        temp_n_batch = self.n_batch
        self.n_batch = n_digits

        if self.vampprior_bool:

            self.n_batch = deepcopy(self.n_pseudo_inputs)

            self.u = self.pseudo_means(self.input_mean)
            self.q_z_given_u_hidden_1st = self.q_z_given_x_hidden_layer[-1](self.u)
            self.q_z_given_u_hidden_2nd = self.q_z_given_x_and_z_hidden_layer[-1](self.q_z_given_u_hidden_1st)
            self.z_given_u_mean = self.q_z_mean_layer[-1](self.q_z_given_u_hidden_2nd)
            self.z_given_u_log_var = self.q_z_log_var_layer[-1](self.q_z_given_u_hidden_2nd)

            self.z_L = self.reparameterization(self.z_given_u_mean, self.z_given_u_log_var)

            if self.weighted_vp_bool:
                self.z_L = self.z_L[torch.argsort(self.w_pseudo_layer.linear_layer.weight, descending=True).squeeze()]

            self.z_L = self.z_L[:n_digits]

            self.n_batch = deepcopy(n_digits)
        else:
            self.z_L = self.reparameterization(torch.zeros(self.z_dim, device=self.device),
                                               torch.zeros(self.z_dim,
                                                           device=self.device))  ### variance is in log-space

        self.q_z_layer = [self.z_L]
        self.decoder()

        self.n_batch = deepcopy(temp_n_batch)

        return self.p_x_mean

    def marginal_LL(self, X, n_samples=5000, batch_size=None):

        if batch_size is not None:
            temp_n_batch = self.n_batch
            self.n_batch = batch_size

        self.marg_ll = []

        for i in range(X.size(0)):

            self.negative_loss_lst = []

            for j in range(int(n_samples / self.n_batch)):
                x = X[i].repeat(self.n_batch, 1)

                self.forward(x)
                self.compute_loss(x)

                self.negative_loss_lst.append(-self.loss.cpu().data.numpy())

            self.marg_ll.append(logsumexp(self.negative_loss_lst) - np.log(len(self.negative_loss_lst)))

        if batch_size is not None:
            self.n_batch = temp_n_batch

        return self.marg_ll

    def measure_dead_units(self, X, batch_size=None):

        if batch_size is not None:
            temp_n_batch = self.n_batch
            self.n_batch = batch_size

        self.z_dead_units_per_layer = np.zeros((self.layers, self.z_dim))
        for n in range(int(X.size(0) / self.n_batch)):
            x = X[n * self.n_batch: (n + 1) * self.n_batch]
            self.forward(x)

            ### Remember the means are saved in reversed format in self.en_q_z_mean_layer
            for en_layer, layer in enumerate(reversed(range(self.layers))):
                self.z_dead_units_per_layer[en_layer] += np.var(self.en_q_z_mean_layer[layer].cpu().data.numpy(),
                                                                axis=0)

        self.z_dead_units_per_layer /= (X.size(0) / self.n_batch)

        if batch_size is not None:
            self.n_batch = temp_n_batch

        return self.z_dead_units_per_layer

"""## Training"""

def training_model(model_name, train_data, valid_data, device, layers, x_dim, h_dim, z_dim, n_batch, eta, epochs,
                   n_pseudo_inputs, early_stopping, warm_up, weighted_vp_bool, He_bool, Lambda, plot_show_bool=False):
    vae = Model(x_dim, h_dim, z_dim, n_batch, device, n_pseudo_inputs, layers, weighted_vp_bool, He_bool, Lambda=Lambda)
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
#        print(vae.w_pseudo_layer.bias)
        ### Warm-up (begin)
        if epoch == 0 and warm_up > 0:
            vae.beta = 0
        elif epoch >= warm_up:
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
            #vae.loss.backward(retain_graph=True)
            vae.loss.backward()
            #print(vae.w_pseudo_layer.linear_layer.weight.grad)
            optimizer.step()

            #if weighted_vp_bool:
                #print('BEFORE SOFTMAX: ',vae.w_pseudo_layer.weight)
                #vae.w_pseudo_layer.weight = nn.Parameter(vae.softmax_layer(vae.w_pseudo_layer.weight))
                #print('AFTER SOFTMAX: ',vae.w_pseudo_layer.weight)
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

        tqdm_epochs.set_description(
            "{} Epoch:{} Training Loss:{} Validation Loss:{}, W:{}".format(torch.sum(vae.w_pseudo_layer.linear_layer.weight), epoch, train_loss_lst[-1], valid_loss_lst[-1], vae.w_pseudo_layer.linear_layer.weight))

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

"""## Testing"""

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

"""## Generator"""

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

"""# Run

## Run Me
"""

def update_log_file(filename, text):
    with open(filename, "a") as fl:
        fl.write(text)


def run_me(USE_CUDA, layers, n_pseudo_inputs, x_dim, h_dim, z_dim, n_batch, eta, epochs, early_stopping,
           warm_up, device, train_data, valid_data, test_data, He_bool, weighted_vp_bool, Lambda):
    text = "\n{}\nmodel_name:{}\t\tLayers:{}\t\tpseudo_inputs:{}\t\tGPU:{}\t\teta:{}\t\twarm_up:{}\n".format(
        str(datetime.now()), model_name, layers, n_pseudo_inputs, USE_CUDA, eta, warm_up)
    text += "early_stopping:{}\t\tbatch_size:{}\t\tvectorized_input_dim:{}\t\tlatent_dim:{}\t\thidden_units:{}\n".format(
        early_stopping, n_batch, x_dim, z_dim, h_dim)
    text += "epochs:{}\t\tHe_initialization:{}\t\tweighted_VP:{}\n".format(epochs, He_bool, bool(weighted_vp_bool))
    text += "\nStart Training at {}".format(str(datetime.now()))
    print(text)
    update_log_file(os.path.join("Results", model_name, "logfile.txt"), text)

    ############################## TRAINING ##############################
    model, train_loss_lst, valid_loss_lst = training_model(model_name, train_data, valid_data, device, layers,
                                                           x_dim, h_dim, z_dim, n_batch, eta, epochs,
                                                           n_pseudo_inputs, early_stopping, warm_up, weighted_vp_bool,
                                                           He_bool, plot_show_bool=False,Lambda=Lambda)
    ############################## Training ##############################

    text = "\nEnd of training at {}\tbest_validation_loss:{}\tbest_epoch:{}\n".format(
        str(datetime.now()), np.min(valid_loss_lst[warm_up:]), len(valid_loss_lst) - early_stopping)
    print(text)
    update_log_file(os.path.join("Results", model_name, "logfile.txt"), text)

    text = "\nStart generating digits by sampling from the prior (time:{})".format(str(datetime.now()))
    print(text)
    update_log_file(os.path.join("Results", model_name, "logfile.txt"), text)

    ############################## GENERATING ##############################
    generated_samples = generation_model(model, model_name)
    ############################## Generating ##############################

    text = "\nEnd of generating digits by sampling from the prior (time:{})\n".format(str(datetime.now()))
    print(text)
    update_log_file(os.path.join("Results", model_name, "logfile.txt"), text)

    text = "\nStart measureing the active units for the validation dataset at {}\n".format(str(datetime.now()))
    print(text)
    update_log_file(os.path.join("Results", model_name, "logfile.txt"), text)

    X = Variable(torch.tensor(valid_data).float().to(device))

    ############################## ACTIVE UNITS (VALIDATION) ##############################
    val_active_units_per_layer = measure_active_units_model(X, model)
    ############################## Active Units (Validation) ##############################

    text = "\nEnd of measuring the active units for the validation dataset at {}".format(str(datetime.now()))
    text += "\nActive units per layer {}\n".format(str(val_active_units_per_layer)[1:-1])
    text += "\nStart measureing the active units for the testing dataset at {}\n".format(str(datetime.now()))
    print(text)
    update_log_file(os.path.join("Results", model_name, "logfile.txt"), text)

    X = Variable(torch.tensor(test_data).float().to(device))

    ############################## ACTIVE UNITS (TESTING) ##############################
    test_active_units_per_layer = measure_active_units_model(X, model)
    ############################## Active Units (Testing) ##############################

    text = "\nEnd of measuring the active units for the testing dataset at {}".format(str(datetime.now()))
    text += "\nActive units per layer {}\n".format(str(test_active_units_per_layer)[1:-1])

    text += '\nStart measuring the averaged marginal log-likelihood for the validation dataset at {}'.format(
        str(datetime.now()))
    print(text)
    update_log_file(os.path.join("Results", model_name, "logfile.txt"), text)

    X = Variable(torch.tensor(valid_data).float().to(device))

    ############################## MARGINAL LL (VALIDATION) ##############################
    val_avg_marginal_LL = measure_marginal_LL_model(X, model, model_name, "validation")
    ############################## Marginal LL (Validation) ##############################

    text = "\nEnd of measuring the averaged marginal log-likelihood for the validation dataset at {}\n".format(
        str(datetime.now()))
    text += "The averaged marginal log-likelihood for the validation dataset is {}\n".format(
        val_avg_marginal_LL)

    text += '\nStart measuring the averaged marginal log-likelihood for the testing dataset at {}'.format(
        str(datetime.now()))
    print(text)
    update_log_file(os.path.join("Results", model_name, "logfile.txt"), text)

    X = Variable(torch.tensor(test_data).float().to(device))

    ############################## MARGINAL LL (TESTING) ##############################
    test_avg_marginal_LL = measure_marginal_LL_model(X, model, model_name, "test")
    ############################## Marginal LL (Testing) ##############################

    text = "\nEnd of measuring the averaged marginal log-likelihood for the testing dataset at {}\n".format(
        str(datetime.now()))
    text += "The averaged marginal log-likelihood for the testing dataset is {}\n\n".format(
        test_avg_marginal_LL)
    print(text)
    update_log_file(os.path.join("Results", model_name, "logfile.txt"), text)

"""## Read Data"""

if __name__ == "__main__":
    ### Prepraing data
    download_bool = False  # True if you need to download all the data (especially for the first time)
    if download_bool:
        download_binarirzed_mnist()  # Downloading data
        read_original_data_and_write_pickled_data_gzip()  # Creating the numpy arrays and save the arrays to pickles
    train_data, valid_data, test_data = read_pickled_data_gzip()  # Read the data from the saved (compressed) pickles
    print("\nShapes for train:{} valid:{} test:{}\n".format(train_data.shape, valid_data.shape, test_data.shape))

    print("Using Torch Version:{}".format(torch.__version__))
    USE_CUDA = torch.cuda.is_available()  # Do you have a GPU or CPU
    device = torch.device("cuda" if USE_CUDA else "cpu")  #
    print("Device in use {}".format(device))
    if USE_CUDA:  # If you have GPU, then how many GPUs do you have
        num_of_gpu = torch.cuda.device_count()  # How many GPUs?
        print("Available GPUs:{}".format(num_of_gpu))
        for n_gpu in range(num_of_gpu):
            print("GPU:{} is a '{}'".format(n_gpu, torch.cuda.get_device_name(n_gpu)))  # Which GPU do you have

    layers = 3
    N = train_data.shape[0]  # Number of training data
    x_dim = train_data.shape[1]
    h_dim = 300
    z_dim = 40
    n_batch = 100
    eta = 5e-4  # Learning rate
    epochs = int(4e3)
    n_pseudo_inputs = 500  # Number of pseudo inputs (500)
    early_stopping = 50  # How many epochs should I wait? (if None then I do not apply early stopping)
    warm_up = 100
    He_bool = True
    weighted_vp_bool = False

    layers = 4
    N = train_data.shape[0]  # Number of training data
    x_dim = train_data.shape[1]
    h_dim = 300
    z_dim = 40
    n_batch = 100
    eta = 1e-4  # Learning rate
    epochs = int(1e5)
    n_pseudo_inputs = 500  # Number of pseudo inputs (500)
    early_stopping = 50  # How many epochs should I wait? (if None then I do not apply early stopping)
    warm_up = 100
    He_bool = True
    weighted_vp_bool = True
    Lambda = 1

    model_name = ("VAE_" if layers == 1 else "HVAE_L_{}_".format(layers)) + (
        "SG_" if n_pseudo_inputs == 0 else "VP_") + ("WG_" if bool(weighted_vp_bool) else "") + (
                     "Wu_" if warm_up > 0 else "") + ("He" if He_bool else "Xa")  + "_new_Weighted"

    if not os.path.exists(os.path.join("Results", model_name)):
        os.makedirs(os.path.join("Results", model_name))
    else:
        print("Skipping the model:{}".format(model_name))


    run_me(USE_CUDA, layers, n_pseudo_inputs, x_dim, h_dim, z_dim, n_batch, eta, epochs, early_stopping,
           warm_up, device, train_data, valid_data, test_data, He_bool, weighted_vp_bool, Lambda)



