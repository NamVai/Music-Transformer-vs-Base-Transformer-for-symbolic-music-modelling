import math

import torch


def nll_from_loss(loss: torch.Tensor) -> float:
    """Convert a loss tensor to scalar NLL."""
    return float(loss.item())


def perplexity_from_nll(nll: float) -> float:
    """Compute perplexity from NLL."""
    return float(math.exp(nll))
