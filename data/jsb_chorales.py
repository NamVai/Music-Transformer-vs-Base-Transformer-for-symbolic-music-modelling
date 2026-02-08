import os
import pickle
from typing import Dict, List, Tuple

import requests
import torch


JSB_URL = (
    "https://raw.githubusercontent.com/czhuang/JSB-Chorales-dataset/master/"
    "jsb-chorales-16th.pkl"
)


def _download_jsb(cache_path: str) -> None:
    """Download the JSB chorales pickle file into the cache if missing."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        return
    resp = requests.get(JSB_URL, timeout=60)
    resp.raise_for_status()
    with open(cache_path, "wb") as f:
        f.write(resp.content)


def _extract_satb_step(step, rest_pitch: int) -> Tuple[int, int, int, int]:
    """Extract S/A/T/B pitches for one timestep, using rest_pitch for missing voices."""
    # JSB chorales have 4 voices; we map each timestep to SATB notes.
    # When a voice is missing, use a rest token.
    if hasattr(step, "shape"):
        # step is a piano-roll vector (128,)
        active = list((step > 0.5).nonzero()[0])
    else:
        active = list(step)
    active = sorted(active)
    # Heuristic split: lowest -> Bass, highest -> Soprano, remaining -> Alto/Tenor
    if len(active) == 0:
        return rest_pitch, rest_pitch, rest_pitch, rest_pitch
    if len(active) == 1:
        n = active[0]
        return n, rest_pitch, rest_pitch, n
    if len(active) == 2:
        b, s = active[0], active[-1]
        return s, rest_pitch, rest_pitch, b
    if len(active) == 3:
        b, t, s = active[0], active[1], active[2]
        return s, t, rest_pitch, b
    b = active[0]
    s = active[-1]
    a = active[-2]
    t = active[1]
    return s, a, t, b


def _encode_sequence(
    seq,
    note_on_base: int,
    time_shift_id: int,
    bar_id: int,
    phrase_id: int,
    rest_pitch: int,
    min_midi: int = 21,
    max_midi: int = 108,
    bar_every: int = 16,
    phrase_every_bars: int = 4,
) -> List[int]:
    """Encode a JSB sequence into event tokens with bar/phrase markers."""
    # Event-based SATB: S_NOTE, A_NOTE, T_NOTE, B_NOTE, TIME_SHIFT
    tokens: List[int] = []
    num_steps = len(seq)
    for i in range(num_steps):
        if bar_every > 0 and i % bar_every == 0:
            tokens.append(bar_id)
            if phrase_every_bars > 0 and (i // bar_every) % phrase_every_bars == 0:
                tokens.append(phrase_id)
        step = seq[i]
        s, a, t, b = _extract_satb_step(step, rest_pitch)
        for pitch in (s, a, t, b):
            if pitch == rest_pitch:
                tokens.append(note_on_base + (rest_pitch - rest_pitch))
            else:
                pitch = max(min_midi, min(max_midi, pitch))
                tokens.append(note_on_base + (pitch - rest_pitch))
        tokens.append(time_shift_id)
    return tokens


def load_jsb_chorales(
    data_dir: str,
    block_size: int = 512,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Load JSB Chorales, tokenize to events, and return split tensors and vocab."""
    cache_path = os.path.join(data_dir, "cache", "jsb-chorales-16th.pkl")
    _download_jsb(cache_path)
    with open(cache_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    min_midi = 21
    max_midi = 108
    rest_pitch = 20  # one below min_midi
    note_on_base = 0
    note_on_count = max_midi - rest_pitch + 1
    time_shift_id = note_on_base + note_on_count
    bar_id = time_shift_id + 1
    phrase_id = time_shift_id + 2
    bos_id = time_shift_id + 3
    eos_id = time_shift_id + 4
    pad_id = time_shift_id + 5
    vocab_size = time_shift_id + 6

    vocab = {
        "note_on_base": note_on_base,
        "time_shift_id": time_shift_id,
        "bar_id": bar_id,
        "phrase_id": phrase_id,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "pad_id": pad_id,
        "vocab_size": vocab_size,
        "min_midi": min_midi,
        "rest_pitch": rest_pitch,
    }

    def split_to_tensor(split: str) -> torch.Tensor:
        pieces = data[split]
        tokens_all: List[int] = []
        for seq in pieces:
            tokens_all.append(bos_id)
            tokens_all.extend(
                _encode_sequence(
                    seq,
                    note_on_base,
                    time_shift_id,
                    bar_id,
                    phrase_id,
                    rest_pitch,
                    min_midi,
                    max_midi,
                )
            )
            tokens_all.append(eos_id)
        # Chunk into fixed blocks for fast training
        tokens = torch.tensor(tokens_all, dtype=torch.long)
        num_blocks = (tokens.numel() - 1) // block_size
        trimmed = tokens[: num_blocks * block_size + 1]
        x = trimmed[:-1].view(num_blocks, block_size)
        y = trimmed[1:].view(num_blocks, block_size)
        return torch.stack([x, y], dim=1)  # (N, 2, L)

    split_tensors = {
        "train": split_to_tensor("train"),
        "valid": split_to_tensor("valid"),
        "test": split_to_tensor("test"),
    }
    return split_tensors, vocab
