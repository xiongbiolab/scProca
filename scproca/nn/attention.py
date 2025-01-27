from torch import Tensor
import torch.nn.functional as F


def masked_cross_attention(
        query: Tensor,
        reference: Tensor,
        valid: Tensor) -> Tensor:

    return F.scaled_dot_product_attention(reference, reference[valid], query[valid])
