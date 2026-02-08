import time
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.music_transformer import MusicTransformer
from utils.metrics import nll_from_loss, perplexity_from_nll


def _make_loader(
    split_tensor: torch.Tensor, batch_size: int, shuffle: bool
) -> DataLoader:
    """Create a DataLoader for a split."""
    x = split_tensor[:, 0]
    y = split_tensor[:, 1]
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Compute average loss on a split."""
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss.item())
    return float(sum(losses) / max(1, len(losses)))


def train_music_transformer(
    splits: Dict[str, torch.Tensor],
    vocab: Dict[str, int],
    device: torch.device,
    config: Dict,
) -> Tuple[MusicTransformer, Dict[str, list]]:
    """Train a Music Transformer with relative positional attention."""
    model = MusicTransformer(
        vocab_size=vocab["vocab_size"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
        max_relative_position=config["max_relative_position"],
    ).to(device)

    train_loader = _make_loader(
        splits["train"], config["batch_size"], shuffle=True
    )
    valid_loader = _make_loader(
        splits["valid"], config["batch_size"], shuffle=False
    )

    criterion = nn.CrossEntropyLoss(ignore_index=vocab["pad_id"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    history = {"train": [], "val": [], "val_ppl": [], "epoch_time": []}
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        epoch_losses = []
        t0 = time.time()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            epoch_losses.append(loss.item())

        train_nll = nll_from_loss(torch.tensor(epoch_losses).mean())
        val_nll = _evaluate(model, valid_loader, criterion, device)
        val_ppl = perplexity_from_nll(val_nll)
        history["train"].append(train_nll)
        history["val"].append(val_nll)
        history["val_ppl"].append(val_ppl)
        history["epoch_time"].append(time.time() - t0)
        run_label = config.get("run_label")
        prefix = f"[{run_label}][Relative]" if run_label else "[Relative]"
        print(
            f"{prefix} Epoch {epoch}/{config['epochs']} "
            f"Train NLL: {train_nll:.4f} "
            f"Val NLL: {val_nll:.4f} "
            f"Val PPL: {val_ppl:.2f} "
            f"Time: {history['epoch_time'][-1]:.1f}s"
        )

    return model, history
