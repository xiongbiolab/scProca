import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from scproca.nn.base import MLP
import torch.nn.functional as F


class RNADecoder(nn.Module):
    def __init__(self, d_in, d_out, d_hids, d_cat, dropout=0.2, norm="BatchNorm", activation="relu",
                 distribution="ZINB"):
        super(RNADecoder, self).__init__()
        self.distribution = distribution
        if self.distribution == "ZINB":
            self.log_theta = nn.Parameter(torch.randn(d_out))
            self.mlp_scale = MLP(d_in=d_in, d_out=d_hids[-1], d_hids=d_hids[:-1], d_cat=d_cat,
                                 dropout_hid=dropout, dropout_out=dropout,
                                 norm_hid=norm, norm_out=norm,
                                 activation_hid=activation, activation_out=activation)
            self.linear_logit_scale = nn.Linear(d_in + d_hids[-1] + d_cat, d_out)
            self.mlp_dropout = MLP(d_in=d_in, d_out=d_hids[-1], d_hids=d_hids[:-1], d_cat=d_cat,
                                   dropout_hid=dropout, dropout_out=dropout,
                                   norm_hid=norm, norm_out=norm,
                                   activation_hid=activation, activation_out=activation)
            self.linear_logit_dropout = nn.Linear(d_in + d_hids[-1] + d_cat, d_out)
        if self.distribution == "NB":
            self.log_theta = nn.Parameter(torch.randn(d_out))
            self.mlp_scale = MLP(d_in=d_in, d_out=d_hids[-1], d_hids=d_hids[:-1], d_cat=d_cat,
                                 dropout_hid=dropout, dropout_out=dropout,
                                 norm_hid=norm, norm_out=norm,
                                 activation_hid=activation, activation_out=activation)
            self.linear_logit_scale = nn.Linear(d_in + d_hids[-1] + d_cat, d_out)

    def forward(self, x, cat, size):
        if self.distribution == "ZINB":
            mlp_scale = self.mlp_scale(x, cat)
            mean = F.softmax(self.linear_logit_scale(torch.cat((x, mlp_scale, cat), dim=-1)), dim=-1) * size
            mlp_dropout = self.mlp_dropout(x, cat)
            logit_dropout = self.linear_logit_dropout(torch.cat((x, mlp_dropout, cat), dim=-1))
            return mean, logit_dropout
        if self.distribution == "NB":
            mlp_scale = self.mlp_scale(x, cat)
            mean = F.softmax(self.linear_logit_scale(torch.cat((x, mlp_scale, cat), dim=-1)), dim=-1) * size
            return (mean,)

    def parameters_shared(self):
        return (torch.exp(self.log_theta),)


class ADTDecoder(nn.Module):
    def __init__(self, d_in, d_out, d_hids, d_cat, prior_parameters, dropout=0.2, norm="BatchNorm", activation="relu",
                 distribution="MixtureNB"):
        super(ADTDecoder, self).__init__()
        self.distribution = distribution
        if self.distribution == "MixtureNB":
            self.log_theta = nn.Parameter(2 * torch.randn(d_out))
            self.prior_mean_log_beta = torch.nn.Parameter(
                torch.from_numpy(prior_parameters["init_background_mean_adt"].astype(np.float32))
            )
            self.prior_log_std_log_beta = torch.nn.Parameter(
                torch.log(torch.from_numpy(prior_parameters["init_background_std_adt"].astype(np.float32)))
            )
            self.mlp_beta = MLP(d_in=d_in, d_out=d_hids[-1], d_hids=d_hids[:-1], d_cat=d_cat,
                                dropout_hid=dropout, dropout_out=dropout,
                                norm_hid=norm, norm_out=norm,
                                activation_hid=activation, activation_out=activation)
            self.linear_mean_log_beta = nn.Linear(d_in + d_hids[-1] + d_cat, d_out)
            self.linear_std_log_beta = nn.Linear(d_in + d_hids[-1] + d_cat, d_out)
            self.mlp_alpha = MLP(d_in=d_in, d_out=d_hids[-1], d_hids=d_hids[:-1], d_cat=d_cat,
                                 dropout_hid=dropout, dropout_out=dropout,
                                 norm_hid=norm, norm_out=norm,
                                 activation_hid=activation, activation_out=activation)
            self.linear_alpha = nn.Linear(d_in + d_hids[-1] + d_cat, d_out)
            self.mlp_dropout = MLP(d_in=d_in, d_out=d_hids[-1], d_hids=d_hids[:-1], d_cat=d_cat,
                                   dropout_hid=dropout, dropout_out=dropout,
                                   norm_hid=norm, norm_out=norm,
                                   activation_hid=activation, activation_out=activation)
            self.linear_logit_dropout = nn.Linear(d_in + d_hids[-1] + d_cat, d_out)
        if self.distribution == "NB":
            self.register_buffer(
                "library_log_means", torch.from_numpy(prior_parameters["library_log_means"]).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(prior_parameters["library_log_vars"]).float()
            )
            self.log_theta = nn.Parameter(2 * torch.randn(d_out))
            self.mlp_scale = MLP(d_in=d_in, d_out=d_hids[-1], d_hids=d_hids[:-1], d_cat=d_cat,
                                 dropout_hid=dropout, dropout_out=dropout,
                                 norm_hid=norm, norm_out=norm,
                                 activation_hid=activation, activation_out=activation)
            self.linear_logit_scale = nn.Linear(d_in + d_hids[-1] + d_cat, d_out)

    def forward(self, x, cat, size=None, valid_adt=None, mode="train"):
        if self.distribution == "MixtureNB":
            mlp_beta = self.mlp_beta(x, cat)
            mean_log_beta = self.linear_mean_log_beta(torch.cat((x, mlp_beta, cat), dim=-1))
            std_log_beta = F.softplus(self.linear_std_log_beta(torch.cat((x, mlp_beta, cat), dim=-1))) + 1e-8
            beta = torch.exp(torch.clamp(Normal(mean_log_beta, std_log_beta).rsample(), max=12))
            mlp_alpha = self.mlp_alpha(x, cat)
            alpha = torch.relu(self.linear_alpha(torch.cat((x, mlp_alpha, cat), dim=-1))) + 1 + 1e-8
            mlp_dropout = self.mlp_dropout(x, cat)
            logit_dropout = self.linear_logit_dropout(torch.cat((x, mlp_dropout, cat), dim=-1))
            return beta, alpha, logit_dropout, mean_log_beta, std_log_beta
        if self.distribution == "NB":
            mlp_scale = self.mlp_scale(x, cat)
            if mode == "imputation":
                mean_log_size = cat @ self.library_log_means
                vars_log_size = cat @ self.library_log_vars
                size_prior = torch.exp(torch.clamp(Normal(mean_log_size, vars_log_size.sqrt()).rsample(), max=12))
                size = size * valid_adt.unsqueeze(-1).float() + size_prior * (1 - valid_adt.unsqueeze(-1).float())
            mean = F.softmax(self.linear_logit_scale(torch.cat((x, mlp_scale, cat), dim=-1)), dim=-1) * size
            return (mean,)

    def parameters_shared(self):
        if self.distribution == "MixtureNB":
            return torch.exp(self.log_theta), self.prior_mean_log_beta, torch.exp(self.prior_log_std_log_beta)
        if self.distribution == "NB":
            return (torch.exp(self.log_theta),)
