import torch
import torch.nn as nn
import torch.nn.functional as F

from scproca.nn.base import Classifier
from scproca.nn.encoder import Encoder, NormalEncoder
from scproca.nn.decoder import RNADecoder, ADTDecoder
from scproca.nn.attention import masked_cross_attention
from scproca.nn.loss import NegativeBinomialLoss, ZeroInflatedNegativeBinomialLoss, MixtureNegativeBinomialLoss

from typing import Literal

import logging

logger = logging.getLogger("scProca")


class scProca_VAE(nn.Module):

    def __init__(
            self,
            d_rna: int,
            d_adt: int,
            n_batch: int,
            prior_parameters: dict,
            d_latent: int = 20,
            d_hidden: tuple = (256, 256),
            activation: str = "relu",
            norm: str = "BatchNorm",
            dropout: float = 0.2,
            distribution_rna: Literal["ZINB", "NB"] = "NB",
            distribution_adt: Literal["MixtureNB", "NB"] = "MixtureNB",
    ):
        super().__init__()
        self.d_rna = d_rna
        self.d_adt = d_adt
        self.n_batch = n_batch
        assert self.n_batch > 1, "n_batch must be greater than 1"
        self.d_latent = d_latent

        self.encoder_rna = Encoder(
            d_rna, d_latent, d_hids=d_hidden, d_cat=n_batch, activation=activation, norm=norm,
            dropout=dropout
        )
        self.decoder_rna = RNADecoder(
            d_latent, d_rna, d_hids=d_hidden[::-1], d_cat=n_batch, activation=activation, norm=norm,
            dropout=dropout, distribution=distribution_rna
        )

        self.encoder_adt = Encoder(
            d_adt, d_latent, d_hids=d_hidden, d_cat=n_batch, activation=activation, norm=norm,
            dropout=dropout
        )
        self.decoder_adt = ADTDecoder(
            d_latent, d_adt, d_hids=d_hidden[::-1], d_cat=n_batch, activation=activation, norm=norm,
            dropout=dropout, distribution=distribution_adt, prior_parameters=prior_parameters)

        self.integrate = NormalEncoder(
            self.d_latent * 2, d_latent, d_hids=(d_hidden[-1],), activation=activation, norm=norm, dropout=dropout)

        self.classifier_embedding_rna = Classifier(
            self.d_latent, n_batch, d_hids=(32, 32), activation=activation, norm=norm)
        self.classifier_embedding_adt = Classifier(
            self.d_latent, n_batch, d_hids=(32, 32), activation=activation, norm=norm)
        self.classifier_embedding = Classifier(
            self.d_latent, n_batch, d_hids=(32, 32), activation=activation, norm=norm)

        self.distribution_rna = distribution_rna
        self.distribution_adt = distribution_adt

    def encode(self, rna, adt, batch_one_hot, valid_adt):

        size_rna = rna.sum(dim=-1).unsqueeze(1)
        size_adt = adt.sum(dim=-1).unsqueeze(1)
        rna = torch.log(1 + rna)
        embedding_rna = self.encoder_rna(rna, batch_one_hot)
        adt = torch.log(1 + adt)
        embedding_adt = self.encoder_adt(adt, batch_one_hot)
        embedding_adt = masked_cross_attention(query=embedding_adt, reference=embedding_rna, valid=valid_adt.bool())
        embedding = self.integrate(torch.cat([embedding_rna, embedding_adt], dim=-1))
        return embedding, embedding_rna, embedding_adt, size_rna, size_adt

    def decode(self, latent, embedding_rna, embedding_adt, size_rna, size_adt, batch_one_hot, valid_adt,
               also_from_embedding=False, mode="train"):

        parameters_unshared_rna_from_latent = self.decoder_rna(
            latent, batch_one_hot, size_rna)
        parameters_shared_rna = self.decoder_rna.parameters_shared()

        parameters_unshared_adt_from_latent = self.decoder_adt(
            latent, batch_one_hot, size_adt, valid_adt, mode=mode)
        parameters_shared_adt = self.decoder_adt.parameters_shared()

        if also_from_embedding:
            parameters_unshared_rna_from_embedding = self.decoder_rna(
                embedding_rna, batch_one_hot, size_rna)
            parameters_unshared_adt_from_embedding = self.decoder_adt(
                embedding_adt, batch_one_hot, size_adt, mode=mode)
        else:
            parameters_unshared_rna_from_embedding = None
            parameters_unshared_adt_from_embedding = None

        return (parameters_unshared_rna_from_latent,
                parameters_unshared_rna_from_embedding,
                parameters_shared_rna,
                parameters_unshared_adt_from_latent,
                parameters_unshared_adt_from_embedding,
                parameters_shared_adt)

    def reconstruction_loss(
            self,
            rna,
            parameters_unshared_rna_from_latent,
            parameters_unshared_rna_from_embedding,
            parameters_shared_rna,
            adt,
            parameters_unshared_adt_from_latent,
            parameters_unshared_adt_from_embedding,
            parameters_shared_adt,
            valid_adt,
            also_from_embedding
    ):
        theta_rna = parameters_shared_rna[0]
        theta_adt = parameters_shared_adt[0]

        if self.distribution_rna == "NB":
            reconstruction_loss_rna_from_latent = NegativeBinomialLoss(
                rna,
                mu=parameters_unshared_rna_from_latent[0],
                theta=theta_rna
            )
            reconstruction_loss_rna_from_embedding = NegativeBinomialLoss(
                rna,
                mu=parameters_unshared_rna_from_embedding[0],
                theta=theta_rna
            ) if also_from_embedding else 0.0
        if self.distribution_rna == "ZINB":
            reconstruction_loss_rna_from_latent = ZeroInflatedNegativeBinomialLoss(
                rna,
                mu=parameters_unshared_rna_from_latent[0],
                pi=parameters_unshared_rna_from_latent[1],
                theta=theta_rna
            )
            reconstruction_loss_rna_from_embedding = ZeroInflatedNegativeBinomialLoss(
                rna,
                mu=parameters_unshared_rna_from_embedding[0],
                pi=parameters_unshared_rna_from_embedding[1],
                theta=theta_rna
            ) if also_from_embedding else 0.0

        if self.distribution_adt == "NB":
            reconstruction_loss_adt_from_latent = NegativeBinomialLoss(
                adt,
                mu=parameters_unshared_adt_from_latent[0],
                theta=theta_adt
            )
            reconstruction_loss_adt_from_embedding = NegativeBinomialLoss(
                adt,
                mu=parameters_unshared_adt_from_embedding[0],
                theta=theta_adt
            ) if also_from_embedding else 0.0
        if self.distribution_adt == "MixtureNB":
            reconstruction_loss_adt_from_latent = MixtureNegativeBinomialLoss(
                adt,
                beta=parameters_unshared_adt_from_latent[0],
                alpha=parameters_unshared_adt_from_latent[1],
                pi=parameters_unshared_adt_from_latent[2],
                theta=theta_adt
            )
            reconstruction_loss_adt_from_embedding = MixtureNegativeBinomialLoss(
                adt,
                beta=parameters_unshared_adt_from_embedding[0],
                alpha=parameters_unshared_adt_from_embedding[1],
                pi=parameters_unshared_adt_from_embedding[2],
                theta=theta_adt
            ) if also_from_embedding else 0.0

        reconstruction_loss_adt_masked_from_latent = reconstruction_loss_adt_from_latent * valid_adt.float()
        reconstruction_loss_adt_masked_from_embedding = reconstruction_loss_adt_from_embedding * valid_adt.float() if also_from_embedding else 0.0

        return (
                reconstruction_loss_rna_from_latent.mean()
                + (reconstruction_loss_rna_from_embedding.mean() if also_from_embedding else 0.0)
                + reconstruction_loss_adt_masked_from_latent.mean()
                + (reconstruction_loss_adt_masked_from_embedding.mean() if also_from_embedding else 0.0)
        )

    def classifier_loss(
            self, latent, embedding_rna, embedding_adt, batch_one_hot
    ):
        output_classifier = self.classifier_embedding(latent)
        output_classifier_rna = self.classifier_embedding_rna(embedding_rna)
        output_classifier_adt = self.classifier_embedding_adt(embedding_adt)

        classifier_loss = F.cross_entropy(output_classifier, batch_one_hot, reduction='mean')
        classifier_loss_rna = F.cross_entropy(output_classifier_rna, batch_one_hot, reduction='mean')
        classifier_loss_adt = F.cross_entropy(output_classifier_adt, batch_one_hot, reduction='mean')

        return classifier_loss + classifier_loss_rna + classifier_loss_adt
