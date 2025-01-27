import torch
import torch.nn as nn
from scproca.nn.base import MLP


class Encoder(nn.Module):
    def __init__(self, d_in, d_out, d_hids, d_cat, dropout=0.2, norm="BatchNorm", activation="relu"):
        super(Encoder, self).__init__()
        if d_hids is None:
            self.mlp = None
            d_linear = d_in
        else:
            self.mlp = MLP(d_in=d_in, d_out=d_hids[-1], d_hids=d_hids[:-1], d_cat=d_cat,
                           dropout_hid=dropout, dropout_out=dropout,
                           norm_hid=norm, norm_out=norm,
                           activation_hid=activation, activation_out=activation)
            d_linear = d_hids[-1]
        self.linear = nn.Linear(d_linear + d_cat, d_out)

    def forward(self, x, cat):
        if self.mlp is not None:
            x = self.mlp(x, cat)
        return self.linear(torch.cat((x, cat), dim=-1))


class NormalEncoder(nn.Module):
    def __init__(self, d_in, d_out, d_hids, dropout=0.2, norm="BatchNorm", activation="relu"):
        super(NormalEncoder, self).__init__()
        if d_hids is None:
            self.mlp = None
            d_linear = d_in
        else:
            self.mlp = MLP(d_in=d_in, d_out=d_hids[-1], d_hids=d_hids[:-1], d_cat=0,
                           dropout_hid=dropout, dropout_out=dropout,
                           norm_hid=norm, norm_out=norm,
                           activation_hid=activation, activation_out=activation)
            d_linear = d_hids[-1]
        self.mean_linear = nn.Linear(d_linear, d_out)
        self.log_var_linear = nn.Linear(d_linear, d_out)

    def forward(self, x):
        if self.mlp is not None:
            x = self.mlp(x)
        mean = self.mean_linear(x)
        var = torch.exp(self.log_var_linear(x)) + 1e-4
        return mean, var.sqrt()
