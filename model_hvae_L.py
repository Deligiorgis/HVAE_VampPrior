import torch
from torch import nn
from torch.autograd import Variable

from scipy.special import logsumexp

import numpy as np

from mlp_layers_Xa_init import GLU as GLU_Xa
from mlp_layers_Xa_init import Layer as Layer_Xa
from mlp_layers_He_init import GLU as GLU_He
from mlp_layers_He_init import Layer as Layer_He

from copy import deepcopy


class Model(nn.Module):

    def __init__(self, x_dim, h_dim, z_dim, n_batch, device, n_pseudo_inputs, layers, weighted_vp_bool, He_bool):
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

        self.vampprior_bool = False
        if n_pseudo_inputs > 0:
            print("VampPrior \t is used as a prior")
            self.vampprior_bool = True
            self.n_pseudo_inputs = n_pseudo_inputs
            self.input_mean = Variable(torch.eye(self.n_pseudo_inputs, device=self.device), requires_grad=False)
            self.pseudo_means = Layer(self.n_pseudo_inputs, self.x_dim, activation=nn.Hardtanh(min_val=0, max_val=1),
                                      initialization=nn.init.normal_, std=0.01, bias=False)

            if self.weighted_vp_bool:
                print("Weighted VampPrior \t is used")
                self.w_pseudo_layer = nn.Linear(self.n_pseudo_inputs, 1, bias=False)
                nn.init.kaiming_normal_(self.w_pseudo_layer.weight)
                self.softmax_layer = nn.Softmax(dim=1)
                self.w_pseudo_layer.weight = nn.Parameter(self.softmax_layer(self.w_pseudo_layer.weight))

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
            log_N_z = self.log_normal(self.z_u, self.z_given_u_mean, self.z_given_u_log_var)  # (100, 500, 40)
            log_N_z = torch.sum(log_N_z, 2)  # (100, 500)
            log_N_z_max, _ = torch.max(log_N_z, 1)
            N_z = self.w_pseudo_layer(torch.exp(log_N_z - log_N_z_max.unsqueeze(1)))
            self.log_prior = log_N_z_max + torch.log(N_z)

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
        self.loss = self.reconstruction_loss.float() + self.beta * self.kl_loss
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
                self.z_L = self.z_L[torch.argsort(self.w_pseudo_layer.weight, descending=True).squeeze()]

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
