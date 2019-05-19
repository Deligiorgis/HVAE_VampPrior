import torch
from torch import nn


class Layer(nn.Module):

    def __init__(self, input_dim, output_dim, activation=None, initialization=nn.init.kaiming_normal_,
                 mu=None, std=None, bias=True):
        super(Layer_He, self).__init__()

        self.activation = activation
        self.layer_model = nn.Linear(input_dim, output_dim, bias=bias)

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


class GLU(nn.Module):

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
