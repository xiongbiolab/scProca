def compute_kl_weight(
        epoch: int,
        step: int,
        n_epochs_kl_warmup: int | None,
        n_steps_kl_warmup: int | None,
        max_kl_weight: float = 1.0,
        min_kl_weight: float = 0.0,
) -> float:
    slope = max_kl_weight - min_kl_weight
    if n_epochs_kl_warmup:
        if epoch < n_epochs_kl_warmup:
            return slope * (epoch / n_epochs_kl_warmup) + min_kl_weight
    elif n_steps_kl_warmup:
        if step < n_steps_kl_warmup:
            return slope * (step / n_steps_kl_warmup) + min_kl_weight
    return max_kl_weight


class EarlyStopping:
    def __init__(self, patience=45, delta=0.0):

        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, loss):
        if loss + self.delta < self.best_loss:
            self.best_loss = loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            self.early_stop = True

        return self.early_stop
