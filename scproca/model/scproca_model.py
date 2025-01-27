from tqdm import tqdm
from anndata import AnnData
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as kl
from torch.utils.data import DataLoader
from scproca import settings
from scproca.module.scproca_module import scProca_VAE
from scproca.utils.data import data2input, to_one_hot, DS, split_dataset_into_train_valid
from scproca.utils.estimation import get_priors_size, get_init_priors_adt
from scproca.utils.optimization import compute_kl_weight, EarlyStopping
from sklearn.preprocessing import LabelEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Literal, List

import logging

logger = logging.getLogger("scProca")


class scProca:
    """Single-cell model of proteomics with/from transcriptomics using cross-attention.

    Parameters
    ----------
    adata : AnnData
        AnnData object. `adata.X` contains the RNA measurements.
    key_adt : str
        The key used to access ADT measurements stored in `adata.obsm`.
    key_batch : str
        The key used to access batch annotations stored in `adata.obs`.
    key_valid_adt : str
        The key used to access whether the ADT measurements are valid or just placeholders in `adata.obs`.
    d_latent : int, optional (default=20)
        The dimensionality of the latent space.
    distribution_rna : {"ZINB", "NB"}, optional (default="NB")
        The distribution to model RNA data. One of:

        * ``'NB'`` - Negative Binomial distribution
        * ``'ZINB'`` - Zero-Inflated Negative Binomial distribution
    distribution_adt : {"MixtureNB", "NB"}, optional (default="MixtureNB")
        The distribution to model ADT data. One of:

        * ``'NB'`` - Negative Binomial distribution
        * ``'MixtureNB'`` - Mixture of two Negative Binomial distributions
    activation : {"relu", "mish"}, optional (default="mish")
        The activation function used in the neural networks. One of:

        * ``'relu'`` - Rectified Linear Unit
        * ``'mish'`` - Mish activation function
    norm : {"BatchNorm", "LayerNorm"}, optional (default="LayerNorm")
        The type of normalization used in the networks. One of:

        * ``'BatchNorm'`` - Batch normalization
        * ``'LayerNorm'`` - Layer normalization
    dropout : float, optional (default=0.2)
        The dropout rate applied during training.
    d_hidden : tuple of int, optional (default=(256, 256))
        A tuple indicating the number of neurons in each hidden layer.
    pre_to_device : bool, optional (default=True)
        Whether to move the data to the device (e.g., GPU) beforehand to reduce data transfer overhead.
        For large datasets, this should be set to False.

    Examples
    --------
    >>> scproca.settings.seed = seed
    >>> scproca.settings.batch_size = batch_size (default=512)
    >>> scproca.settings.device = index_cuda (None if using 'cpu')
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> batch = adata.obs[key_batch].values.ravel()
    >>> valid_adt = np.array([True] * len(adata))
    >>> valid_adt[adt_not_valid] = False
    >>> adata.obs["valid_adt"] = valid_adt
    >>> scproca = scProca(adata=adata, key_adt=key_adt, key_batch=key_batch, key_valid_adt="valid_adt")
    >>> scproca.train()
    >>> adata.obsm["latent"], adata.obsm["embedding_rna"], adata.obsm["embedding_adt"] = scproca.get_latent_representation()
    >>> adata.obsm["protein_generation"] = scproca.generation(anchor_batch=list_str_anchor_batch)
    """

    def __init__(
            self,
            adata: AnnData,
            key_adt: str,
            key_batch: str,
            key_valid_adt: str,
            d_latent: int = 20,
            distribution_rna: Literal["ZINB", "NB"] = "NB",
            distribution_adt: Literal["MixtureNB", "NB"] = "MixtureNB",
            activation: Literal["relu", "mish"] = "mish",
            norm: Literal["BatchNorm", "LayerNorm"] = "LayerNorm",
            dropout: float = 0.2,
            d_hidden: tuple = (256, 256),
            pre_to_device: bool = True,
    ):
        self.device = settings.device
        self.rna = data2input(adata.X, pre_to_device)
        self.adt = data2input(adata.obsm[key_adt].to_numpy().astype(np.float32), pre_to_device)
        self.valid_adt = data2input(adata.obs[key_valid_adt].to_numpy(), pre_to_device)
        self.batch = adata.obs[key_batch].to_numpy()
        self.n_batch = len(np.unique(self.batch))
        label_encoder = LabelEncoder()
        batch_code = np.array(label_encoder.fit_transform(self.batch))
        self.batch_mapping = dict(zip(np.unique(self.batch), label_encoder.transform(np.unique(self.batch))))
        self.batch_one_hot = data2input(to_one_hot(batch_code, self.n_batch), pre_to_device)
        self.pre_to_device = pre_to_device

        self.distribution_rna = distribution_rna
        self.distribution_adt = distribution_adt

        if self.distribution_adt == "MixtureNB":
            init_background_mean_adt, init_background_std_adt = get_init_priors_adt(
                adata.obsm[key_adt], batch_code, adata.obs[key_valid_adt].to_numpy())
        else:
            init_background_mean_adt, init_background_std_adt = None, None

        if self.distribution_adt == "NB":
            library_log_means, library_log_vars = get_priors_size(
                adata.obsm[key_adt], batch_code)
        else:
            library_log_means, library_log_vars = None, None

        self.module = scProca_VAE(
            d_rna=self.rna.shape[-1],
            d_adt=self.adt.shape[-1],
            n_batch=self.n_batch,
            d_latent=d_latent,
            d_hidden=d_hidden,
            activation=activation,
            norm=norm,
            dropout=dropout,
            distribution_rna=self.distribution_rna,
            distribution_adt=self.distribution_adt,
            prior_parameters={
                "init_background_mean_adt": init_background_mean_adt,
                "init_background_std_adt": init_background_std_adt,
                "library_log_means": library_log_means,
                "library_log_vars": library_log_vars,
            }
        ).to(self.device)
        self.log = {
            "loss_elbo": [],
            "loss_discriminator": [],
        }

    def train(
            self,
            batch_size: int | None = None, lambda_a: float = 30.0,
            adversarial_step=1, epochs=400, lr=4e-3,
            ratio_val: float = 0.1, epochs_warmup: int | None = None,
            steps_warmup: int | None = None,
            bool_also_reconstructed_from_embedding: bool = True,
    ):
        """Trains the model using variational inference.

        Parameters
        ----------
        batch_size : int, optional
            The minibatch size used during training. Can also be specified via `scproca.settings.batch_size`.
        lambda_a : float (default=30.0)
            The coefficient for the adversarial loss.
        adversarial_step : int, optional (default=1)
            The number of steps for adversarial network optimization in each training epoch.
        epochs : int, optional (default=400)
            The maximum number of training epochs.
        lr : float, optional (default=4e-3)
            The learning rate.
        ratio_val : float, optional (default=0.1)
            The proportion of the dataset used as the validation set.
        epochs_warmup : int, optional (default=None)
            The number of epochs to use for warmup.
        steps_warmup : int, optional (default=None)
            The number of steps to use for warmup.
        bool_also_reconstructed_from_embedding : bool, optional (default=True)
            Whether to additionally train the reconstruction loss from the embeddings,
            apart from the latent space reconstruction loss.
        """

        if batch_size is None:
            batch_size = settings.batch_size
        else:
            settings.batch_size = batch_size
        parameters_classifiers = (
                list(self.module.classifier_embedding.parameters())
                + list(self.module.classifier_embedding_rna.parameters())
                + list(self.module.classifier_embedding_adt.parameters())
        )
        optimizer = optim.Adam(
            filter(lambda p: id(p) not in map(id, parameters_classifiers),
                   self.module.parameters()), lr=lr, weight_decay=1e-6, eps=0.01
        )
        optimizer_discriminator = optim.Adam(
            parameters_classifiers, lr=1e-3, weight_decay=1e-6, eps=0.01)

        scheduler = ReduceLROnPlateau(
            optimizer,
            patience=30,
            factor=0.6,
            threshold=0.0,
            min_lr=0,
            threshold_mode="abs",
            verbose=True
        )

        epoch_progress = tqdm(range(epochs), desc='Training Progress', ncols=120)
        dataset = DS(self.rna, self.adt, self.batch_one_hot, self.valid_adt)
        dataset_train, dataset_val = split_dataset_into_train_valid(dataset, batch_size, ratio_val)
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
        dataloader_valid = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
        early_stopping = EarlyStopping(patience=45, delta=0.0)
        step = 0
        steps_warmup = (
            steps_warmup if steps_warmup is not None else int(0.75 * len(dataset))
        )
        for epoch in epoch_progress:
            self.module.train()
            for (_, rna, adt, batch_one_hot, valid_adt) in dataloader_train:
                lambda_kl_epoch = compute_kl_weight(epoch, step, epochs_warmup, steps_warmup)
                lambda_a_epoch = lambda_a + 1.0 - lambda_kl_epoch
                if not self.pre_to_device:
                    rna = rna.to(self.device)
                    adt = adt.to(self.device)
                    batch_one_hot = batch_one_hot.to(self.device)
                    valid_adt = valid_adt.to(self.device)
                (latent_mean, latent_std), embedding_rna, embedding_adt, size_rna, size_adt = self.module.encode(
                    rna, adt, batch_one_hot, valid_adt)
                latent_distribution = Normal(latent_mean, latent_std)
                loss_kl = kl(latent_distribution, Normal(0, 1)).sum(dim=-1).mean() * lambda_kl_epoch
                latent = latent_distribution.rsample()

                for _ in range(adversarial_step):
                    loss_discriminator = self.module.classifier_loss(
                        latent.detach(), embedding_rna.detach(), embedding_adt.detach(), batch_one_hot
                    ) * lambda_a_epoch
                    optimizer_discriminator.zero_grad()
                    loss_discriminator.backward()
                    optimizer_discriminator.step()

                (parameters_unshared_rna_from_latent,
                 parameters_unshared_rna_from_embedding,
                 parameters_shared_rna,
                 parameters_unshared_adt_from_latent,
                 parameters_unshared_adt_from_embedding,
                 parameters_shared_adt) = self.module.decode(
                    latent, embedding_rna, embedding_adt, size_rna, size_adt, batch_one_hot, valid_adt,
                    also_from_embedding=bool_also_reconstructed_from_embedding, mode="train")

                loss_r = self.module.reconstruction_loss(
                    rna, parameters_unshared_rna_from_latent, parameters_unshared_rna_from_embedding,
                    parameters_shared_rna,
                    adt, parameters_unshared_adt_from_latent, parameters_unshared_adt_from_embedding,
                    parameters_shared_adt,
                    valid_adt,
                    also_from_embedding=bool_also_reconstructed_from_embedding
                )
                if self.distribution_adt == "MixtureNB":
                    (mean_log_beta, std_log_beta) = parameters_unshared_adt_from_latent[-2:]
                    (prior_mean_log_beta, prior_std_log_beta) = parameters_shared_adt[-2:]
                    prior_mean_log_beta = F.linear(
                        batch_one_hot, prior_mean_log_beta
                    )
                    prior_std_log_beta = F.linear(
                        batch_one_hot, prior_std_log_beta,
                    )

                    loss_kl_log_beta = (kl(
                        Normal(mean_log_beta, std_log_beta),
                        Normal(prior_mean_log_beta, prior_std_log_beta)
                    ).sum(-1) * valid_adt.float()).mean() * lambda_kl_epoch
                else:
                    loss_kl_log_beta = 0.0

                loss_adversarial = - self.module.classifier_loss(
                    latent, embedding_rna, embedding_adt, batch_one_hot) * lambda_a_epoch
                loss = loss_kl + loss_r + loss_kl_log_beta + loss_adversarial

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step += 2

            self.module.eval()
            with torch.no_grad():
                loss_elbo = 0.0
                loss_discriminator = 0.0
                n_val = 0
                for (_, rna, adt, batch_one_hot, valid_adt) in dataloader_valid:
                    if not self.pre_to_device:
                        rna = rna.to(self.device)
                        adt = adt.to(self.device)
                        batch_one_hot = batch_one_hot.to(self.device)
                        valid_adt = valid_adt.to(self.device)
                    (latent_mean, latent_std), embedding_rna, embedding_adt, size_rna, size_adt = self.module.encode(
                        rna, adt, batch_one_hot, valid_adt)
                    latent_distribution = Normal(latent_mean, latent_std)
                    loss_kl = kl(latent_distribution, Normal(0, 1)).sum(dim=-1).mean()
                    latent = latent_distribution.rsample()
                    (parameters_unshared_rna_from_latent,
                     parameters_unshared_rna_from_embedding,
                     parameters_shared_rna,
                     parameters_unshared_adt_from_latent,
                     parameters_unshared_adt_from_embedding,
                     parameters_shared_adt) = self.module.decode(
                        latent, embedding_rna, embedding_adt, size_rna, size_adt, batch_one_hot, valid_adt,
                        also_from_embedding=bool_also_reconstructed_from_embedding, mode="train"
                    )

                    loss_r = self.module.reconstruction_loss(
                        rna, parameters_unshared_rna_from_latent, parameters_unshared_rna_from_embedding,
                        parameters_shared_rna,
                        adt, parameters_unshared_adt_from_latent, parameters_unshared_adt_from_embedding,
                        parameters_shared_adt,
                        valid_adt,
                        also_from_embedding=bool_also_reconstructed_from_embedding
                    )

                    if self.distribution_adt == "MixtureNB":
                        (mean_log_beta, std_log_beta) = parameters_unshared_adt_from_latent[-2:]
                        (prior_mean_log_beta, prior_std_log_beta) = parameters_shared_adt[-2:]
                        prior_mean_log_beta = F.linear(
                            batch_one_hot, prior_mean_log_beta
                        )
                        prior_std_log_beta = F.linear(
                            batch_one_hot, prior_std_log_beta,
                        )

                        loss_kl_log_beta = (kl(
                            Normal(mean_log_beta, std_log_beta),
                            Normal(prior_mean_log_beta, prior_std_log_beta)
                        ).sum(-1) * valid_adt.float()).mean()
                    else:
                        loss_kl_log_beta = 0.0

                    loss_elbo += loss_kl + loss_r + loss_kl_log_beta

                    loss_discriminator += self.module.classifier_loss(
                        latent, embedding_rna, embedding_adt, batch_one_hot)

                    n_val += 1

                if n_val > 0:
                    loss_elbo /= n_val
                    loss_discriminator /= n_val

                    self.log["loss_elbo"].append((float(epoch), float(loss_elbo.item())))
                    self.log["loss_discriminator"].append((float(epoch), float(loss_discriminator.item())))
                    epoch_progress.set_postfix(
                        loss_elbo=loss_elbo.item(),
                        loss_discriminator=loss_discriminator.item(),
                    )
                    early_stop = early_stopping(loss_elbo.item())
                    scheduler.step(loss_elbo.item())
            if n_val > 0 and early_stop:
                print("Early stopping triggered.")
                break

    @torch.inference_mode()
    def get_latent_representation(self, n_shuffle: int | None = 100):
        """Infers the integrated latent representation, RNA-specific embedding, and ADT-specific embedding for each cell.

        Parameters
        ----------
        n_shuffle : int, optional (default=100)
            The number of repetitions used to estimate the mean representation.

        Returns
        -------
        - **latent** - integrated latent representation
        - **embedding_rna** - RNA-specific embedding representation
        - **embedding_adt** - ADT-specific embedding representation
        """

        batch_size = settings.batch_size
        dataset = DS(self.rna, self.adt, self.batch_one_hot, self.valid_adt)
        self.module.eval()
        sum_latent = np.zeros((len(dataset), self.module.d_latent), dtype=np.float32)
        sum_embedding_rna = np.zeros((len(dataset), self.module.d_latent), dtype=np.float32)
        sum_embedding_adt = np.zeros((len(dataset), self.module.d_latent), dtype=np.float32)
        counts = np.zeros((len(dataset), 1), dtype=np.int32)
        epoch_progress = tqdm(range(n_shuffle), desc='Inference Progress', ncols=120)
        for _ in epoch_progress:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            latent = np.zeros((len(dataset), self.module.d_latent), dtype=np.float32)
            embedding_rna = np.zeros((len(dataset), self.module.d_latent), dtype=np.float32)
            embedding_adt = np.zeros((len(dataset), self.module.d_latent), dtype=np.float32)
            for (idx, rna, adt, batch_one_hot, valid_adt) in dataloader:
                if not self.pre_to_device:
                    rna = rna.to(self.device)
                    adt = adt.to(self.device)
                    batch_one_hot = batch_one_hot.to(self.device)
                    valid_adt = valid_adt.to(self.device)
                (latent_mean, _), embedding_from_rna, embedding_from_adt, _, _ = self.module.encode(
                    rna, adt, batch_one_hot, valid_adt)
                latent[idx.numpy()] = latent_mean.cpu().numpy()
                embedding_rna[idx.numpy()] = embedding_from_rna.cpu().numpy()
                embedding_adt[idx.numpy()] = embedding_from_adt.cpu().numpy()
                counts[idx.numpy(), 0] += 1
            sum_latent = sum_latent + latent
            sum_embedding_rna = sum_embedding_rna + embedding_rna
            sum_embedding_adt = sum_embedding_adt + embedding_adt
        return sum_latent / counts, sum_embedding_rna / counts, sum_embedding_adt / counts

    @torch.inference_mode()
    def generation(self, anchor_batch: str | List[str] | None, n_shuffle: int | None = 100):
        """Generates ADT measurements for each cell.

        Parameters
        ----------
        anchor_batch : str or List[str], optional (default=None)
            The batch or list of batches used to the generated measurements.
            If None, it refers to the original batch of the cells.
        n_shuffle : int, optional (default=100)
            The number of repetitions used to estimate the mean generated measurements.

        Returns
        -------
        - **protein_generation** - generated ADT measurements.
        """

        anchor_batch_list = anchor_batch if isinstance(anchor_batch, list) else [anchor_batch]
        anchor_batch_list = [
            None if anchor_batch is None else self.batch_mapping[anchor_batch] for anchor_batch in anchor_batch_list
        ]
        self.module.eval()
        batch_size = settings.batch_size
        dataset = DS(self.rna, self.adt, self.batch_one_hot, self.valid_adt)
        self.module.eval()
        sum_expression = np.zeros((len(dataset), self.module.d_adt), dtype=np.float32)
        counts_expression = np.zeros((len(dataset), 1), dtype=np.int32)
        epoch_progress = tqdm(range(n_shuffle), desc='Inference Progress', ncols=120)
        for _ in epoch_progress:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            expression = np.zeros((len(dataset), self.module.d_adt), dtype=np.float32)
            for (idx, rna, adt, batch_one_hot, valid_adt) in dataloader:
                for anchor_batch in anchor_batch_list:
                    if not self.pre_to_device:
                        rna = rna.to(self.device)
                        adt = adt.to(self.device)
                        batch_one_hot = batch_one_hot.to(self.device)
                        valid_adt = valid_adt.to(self.device)
                    (latent_mean, _), embedding_from_rna, embedding_from_adt, size_rna, size_adt = self.module.encode(
                        rna, adt, batch_one_hot, valid_adt)

                    if anchor_batch is not None:
                        batch_one_hot = torch.zeros_like(batch_one_hot)
                        batch_one_hot[:, anchor_batch] = torch.ones_like(batch_one_hot[:, anchor_batch])

                    (_, _, _, parameters_unshared_adt, _, _) = self.module.decode(
                        latent_mean, embedding_from_rna, embedding_from_adt, size_rna, size_adt, batch_one_hot,
                        valid_adt, mode="imputation"
                    )

                    if self.distribution_adt == "NB":
                        expression[idx.numpy()] = parameters_unshared_adt[0].cpu().numpy()
                    if self.distribution_adt == "MixtureNB":
                        expression[idx.numpy()] = (
                                torch.sigmoid(parameters_unshared_adt[2]) * parameters_unshared_adt[0]
                                + (
                                        1 - torch.sigmoid(parameters_unshared_adt[2])
                                ) * parameters_unshared_adt[0] * parameters_unshared_adt[1]
                        ).cpu().numpy()
                    counts_expression[idx.numpy(), 0] += 1
            sum_expression = sum_expression + expression
        return sum_expression / counts_expression

    def curve_loss(self, key_loss):
        """Plots the loss curve for the validation dataset during the training process.

        Parameters
        ----------
        key_loss : str, optional (default="loss_elbo")
            The key used to specify which loss to plot. Choices are:

            * ``'loss_elbo'`` - ELBO (Evidence Lower Bound) loss
            * ``'loss_discriminator'`` - Loss for the discriminators
        """

        import matplotlib.pyplot as plt
        plt.figure(figsize=(4, 2))
        plt.plot(np.array(self.log[key_loss])[:, 0] + 1, np.array(self.log[key_loss])[:, 1])
        plt.xlabel('training epoch')
        plt.ylabel(key_loss)
        plt.title('Loss curve')
        plt.show()
