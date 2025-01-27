import torch
import torch.nn.functional as F


def NegativeBinomialLoss(x, mu, theta):
    eps = 1e-8
    log_theta_mu_eps = torch.log(theta + mu + eps)
    res = (
            theta * (torch.log(theta + eps) - log_theta_mu_eps)
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
    )

    return -res.sum(-1)


def ZeroInflatedNegativeBinomialLoss(x, mu, pi, theta):
    eps = 1e-8
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))
    softplus_pi = F.softplus(-pi)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
            -softplus_pi
            + pi_theta_log
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero

    return -res.sum(-1)


def MixtureNegativeBinomialLoss(x, beta, alpha, pi, theta):
    eps = 1e-8
    mu_1 = beta
    mu_2 = beta * alpha
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))  # In this case, we reshape theta for broadcasting

    log_theta_mu_1_eps = torch.log(theta + mu_1 + eps)
    log_theta_mu_2_eps = torch.log(theta + mu_2 + eps)
    lgamma_x_theta = torch.lgamma(x + theta)
    lgamma_theta = torch.lgamma(theta)
    lgamma_x_plus_1 = torch.lgamma(x + 1)

    log_nb_1 = (
            theta * (torch.log(theta + eps) - log_theta_mu_1_eps)
            + x * (torch.log(mu_1 + eps) - log_theta_mu_1_eps)
            + lgamma_x_theta
            - lgamma_theta
            - lgamma_x_plus_1
    )
    log_nb_2 = (
            theta * (torch.log(theta + eps) - log_theta_mu_2_eps)
            + x * (torch.log(mu_2 + eps) - log_theta_mu_2_eps)
            + lgamma_x_theta
            - lgamma_theta
            - lgamma_x_plus_1
    )

    logsumexp = torch.logsumexp(torch.stack((log_nb_1, log_nb_2 - pi)), dim=0)
    softplus_pi = F.softplus(-pi)

    log_mixture_nb = logsumexp - softplus_pi
    return - log_mixture_nb.sum(-1)
