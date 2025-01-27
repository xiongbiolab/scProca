import torch
import torch.nn as nn


def Activation(activation):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "gelu":
        return nn.GELU(approximate="tanh")
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "mish":
        return nn.Mish()
    else:
        return None


def Normalization(normalization, dim):
    if normalization == "LayerNorm":
        return nn.LayerNorm(dim, eps=1e-6)
    elif normalization == "BatchNorm":
        return nn.BatchNorm1d(dim, momentum=0.01, eps=0.001)
    else:
        return None


class MLP(nn.Module):
    def __init__(
            self,
            d_in, d_out, d_hids, d_cat,
            dropout_hid, dropout_out,
            norm_hid, norm_out,
            activation_hid, activation_out,
    ):
        super(MLP, self).__init__()

        self.d_in = d_in
        if isinstance(d_hids, int):
            d_hids = (d_hids, )

        layers = []
        input_dim = d_in
        if d_hids is not None:
            for hidden_dim in d_hids:
                layers.append(nn.Linear(input_dim + d_cat, hidden_dim))
                if norm_hid is not None:
                    layers.append(Normalization(norm_hid, hidden_dim))
                if activation_hid is not None:
                    layers.append(Activation(activation_hid))
                if dropout_hid is not None:
                    layers.append(nn.Dropout(dropout_hid))
                input_dim = hidden_dim
        layers.append(nn.Linear(input_dim + d_cat, d_out))
        if norm_out is not None:
            layers.append(Normalization(norm_out, d_out))
        if activation_out is not None:
            layers.append(Activation(activation_out))
        if dropout_out is not None:
            layers.append(nn.Dropout(dropout_out))

        self.layers = nn.Sequential(*layers)

    def forward(self, x, cat=None):
        for layer in self.layers:
            if isinstance(layer, nn.Linear) and cat is not None:
                x = torch.cat((x, cat), dim=-1)
            x = layer(x)
        return x


class Classifier(nn.Module):
    def __init__(self, d_in, d_out, d_hids, dropout=0.1, norm="BatchNorm", activation="relu"):
        super(Classifier, self).__init__()
        if d_hids is None:
            self.mlp = None
            d_linear = d_in
        else:
            self.mlp = MLP(d_in=d_in, d_out=d_hids[-1], d_hids=d_hids[:-1], d_cat=0,
                           dropout_hid=dropout, dropout_out=dropout,
                           norm_hid=norm, norm_out=norm,
                           activation_hid=activation, activation_out=activation)
            d_linear = d_hids[-1]
        self.linear = nn.Linear(d_linear, d_out)

    def forward(self, x):
        if self.mlp is not None:
            x = self.mlp(x)
        return self.linear(x)
