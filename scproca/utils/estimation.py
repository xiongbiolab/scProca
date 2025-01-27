import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("scProca")


def get_init_priors_adt(adt, batch, bool_valid_adt, n_cells=100):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.mixture import GaussianMixture

    logger.info("Computing empirical prior initialization for protein background.")
    pro_exp = adt.to_numpy() if isinstance(adt, pd.DataFrame) else adt

    batch_avg_mus, batch_avg_scales = [], []
    for b in np.unique(batch):
        num_in_batch = np.sum(batch == b)
        if num_in_batch == 0:
            # the values of these batches will not be used
            batch_avg_mus.append(0)
            batch_avg_scales.append(0.05)
            continue
        batch_pro_exp = pro_exp[batch == b]
        batch_valid_adt = bool_valid_adt[batch == b]

        batch_pro_exp = batch_pro_exp[batch_valid_adt]

        if batch_pro_exp.shape[0] == 0:
            # the values of these batches will not be used
            batch_avg_mus.append(0.0)
            batch_avg_scales.append(0.05)
            continue

        cells = np.random.choice(np.arange(batch_pro_exp.shape[0]), size=n_cells)
        batch_pro_exp = batch_pro_exp[cells]
        gmm = GaussianMixture(n_components=2)
        mus, scales = [], []
        # fit per cell GMM
        for c in batch_pro_exp:
            try:
                gmm.fit(np.log1p(c.reshape(-1, 1)))
            # when cell is all 0
            except ConvergenceWarning:
                mus.append(0)
                scales.append(0.05)
                continue

            means = gmm.means_.ravel()
            sorted_fg_bg = np.argsort(means)
            mu = means[sorted_fg_bg].ravel()[0]
            covariances = gmm.covariances_[sorted_fg_bg].ravel()[0]
            scale = np.sqrt(covariances)
            mus.append(mu)
            scales.append(scale)

        # average distribution over cells
        batch_avg_mu = np.mean(mus)
        batch_avg_scale = np.sqrt(np.sum(np.square(scales)) / (n_cells ** 2))

        batch_avg_mus.append(batch_avg_mu)
        batch_avg_scales.append(batch_avg_scale)

    # repeat prior for each protein
    batch_avg_mus = np.array(batch_avg_mus, dtype=np.float32).reshape(1, -1)
    batch_avg_scales = np.array(batch_avg_scales, dtype=np.float32).reshape(1, -1)
    batch_avg_mus = np.tile(batch_avg_mus, (pro_exp.shape[1], 1))
    batch_avg_scales = np.tile(batch_avg_scales, (pro_exp.shape[1], 1))

    return batch_avg_mus, batch_avg_scales


def get_priors_size(
    data, batch
) -> tuple[np.ndarray, np.ndarray]:
    data = data.to_numpy() if isinstance(data, pd.DataFrame) else data
    n_batch = len(np.unique(batch))
    library_log_means = np.zeros(n_batch)
    library_log_vars = np.ones(n_batch)

    for i_batch in np.unique(batch):
        idx_batch = np.squeeze(batch == i_batch)
        batch_data = data[idx_batch.nonzero()[0]]  # h5ad requires integer indexing arrays.
        sum_counts = batch_data.sum(axis=1)
        masked_log_sum = np.ma.log(sum_counts)
        if np.ma.is_masked(masked_log_sum):
            # the values of these batches will not be used
            library_log_means[i_batch] = 0.0
            library_log_vars[i_batch] = 0.05
            continue
        log_counts = masked_log_sum.filled(0)
        library_log_means[i_batch] = np.mean(log_counts).astype(np.float32)
        library_log_vars[i_batch] = np.var(log_counts).astype(np.float32)

    return library_log_means.reshape(-1, 1), library_log_vars.reshape(-1, 1)
