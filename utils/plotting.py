import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def plot_loss_curves(
    curves: Dict[str, Dict[str, List[float]]], output_path: str
) -> None:
    """Plot validation NLL curves for one or more models."""
    # curves: {"baseline": {"train": [...], "val": [...]}, "relative": {...}}
    plt.figure(figsize=(7, 4))
    for name, splits in curves.items():
        if "val" in splits:
            plt.plot(splits["val"], label=f"{name} val")
    plt.xlabel("Epoch")
    plt.ylabel("NLL")
    plt.title("Validation NLL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def pianoroll_from_tokens(
    tokens: List[int],
    note_on_base: int,
    time_shift_id: int,
    min_midi: int = 21,
    rest_pitch: int = 20,
    max_steps: int = 256,
) -> np.ndarray:
    """Decode tokens into a simple pianoroll matrix (T x 88)."""
    # Decode SATB event tokens into a simple piano-roll (T x 88).
    # We read 4 note tokens per timestep (S, A, T, B), then TIME_SHIFT.
    roll = []
    current_notes = []
    steps = 0
    for tok in tokens:
        if tok == time_shift_id:
            frame = np.zeros(88, dtype=np.float32)
            for n in current_notes:
                if min_midi <= n <= min_midi + 87:
                    frame[n - min_midi] = 1.0
            roll.append(frame)
            current_notes = []
            steps += 1
            if steps >= max_steps:
                break
        elif note_on_base <= tok < time_shift_id:
            pitch = (tok - note_on_base) + rest_pitch
            if pitch != rest_pitch:
                current_notes.append(pitch)
    if not roll:
        return np.zeros((1, 88), dtype=np.float32)
    return np.stack(roll, axis=0)


def save_pianoroll_plot(roll: np.ndarray, output_path: str) -> None:
    """Save a pianoroll image to disk."""
    plt.figure(figsize=(8, 3))
    plt.imshow(roll.T, aspect="auto", origin="lower", cmap="gray_r")
    plt.xlabel("Time step")
    plt.ylabel("Pitch (MIDI 21-108)")
    plt.title("Piano-roll")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_pianoroll_grid(
    rolls: List[np.ndarray],
    rows: int,
    cols: int,
    output_path: str,
    titles: List[str] | None = None,
) -> None:
    """Save a grid of pianoroll images."""
    plt.figure(figsize=(cols * 3.2, rows * 2.4))
    for i, roll in enumerate(rolls[: rows * cols]):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(roll.T, aspect="auto", origin="lower", cmap="gray_r")
        ax.set_xticks([])
        ax.set_yticks([])
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_similarity_heatmap(sim: np.ndarray, output_path: str, title: str) -> None:
    """Save a self-similarity heatmap image."""
    plt.figure(figsize=(4.5, 4.5))
    plt.imshow(sim, aspect="auto", origin="lower", cmap="viridis")
    plt.title(title)
    plt.xlabel("Token index")
    plt.ylabel("Token index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
