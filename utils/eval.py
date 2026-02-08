from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils.metrics import perplexity_from_nll


def _make_loader(split_tensor: torch.Tensor, batch_size: int) -> DataLoader:
    """Create a deterministic DataLoader for evaluation."""
    x = split_tensor[:, 0]
    y = split_tensor[:, 1]
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)


def evaluate_splits(
    model: nn.Module,
    splits: Dict[str, torch.Tensor],
    vocab: Dict[str, int],
    device: torch.device,
    batch_size: int,
) -> Dict[str, Dict[str, float]]:
    """Compute NLL and PPL on train/valid/test splits."""
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["pad_id"])
    metrics: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        for split_name in ["train", "valid", "test"]:
            loader = _make_loader(splits[split_name], batch_size)
            losses = []
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                losses.append(loss.item())
            nll = float(sum(losses) / max(1, len(losses)))
            metrics[split_name] = {
                "nll": nll,
                "ppl": perplexity_from_nll(nll),
            }
    return metrics
