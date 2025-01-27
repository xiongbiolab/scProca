import numpy as np
import scipy
from scproca import settings
from torch.utils.data import Dataset, Subset


def data2input(data, to_device=False):
    import torch

    if scipy.sparse.issparse(data):
        data = data.toarray()
    if not isinstance(data, torch.Tensor):
        data = torch.IntTensor(data) if str(data.dtype).startswith("int") else torch.FloatTensor(data)

    if to_device:
        data = data.to(settings.device)

    return data


def to_one_hot(labels, num_classes):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


class DS(Dataset):
    def __init__(self, rna, adt, batch_one_hot, valid_adt):
        self.rna = rna
        self.adt = adt
        self.batch_one_hot = batch_one_hot
        self.valid_adt = valid_adt

    def __len__(self):
        return len(self.rna)

    def __getitem__(self, idx):
        rna = self.rna[idx]
        adata = self.adt[idx]
        batch_one_hot = self.batch_one_hot[idx]
        valid_adt = self.valid_adt[idx]
        return idx, rna, adata, batch_one_hot, valid_adt


def split_dataset_into_train_valid(dataset, batch_size, ratio_val):
    n = len(dataset)

    n_val = int(n * ratio_val)
    n_val = (n_val // batch_size) * batch_size

    index = np.arange(n)
    np.random.shuffle(index)

    index_val = index[:n_val]
    index_train = index[n_val:]

    dataset_train = Subset(dataset, index_train)
    dataset_val = Subset(dataset, index_val)

    return dataset_train, dataset_val
