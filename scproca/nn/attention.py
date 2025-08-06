import torch
from torch import Tensor
import torch.nn.functional as F


def masked_cross_attention(
        query: Tensor,
        reference: Tensor,
        valid: Tensor) -> Tensor:
    return F.scaled_dot_product_attention(reference, reference[valid], query[valid])


def masked_kNN_average(
        query: Tensor,
        reference: Tensor,
        valid: Tensor,
        k: int = 15) -> Tensor:
    d = torch.norm(reference.unsqueeze(1) - reference[valid].unsqueeze(0), p=2, dim=-1)
    _, top_k_indices = d.topk(k, dim=-1, largest=False, sorted=False)
    return query[valid][top_k_indices].mean(dim=1)

