import csv
import os
import zipfile
from typing import Dict, List, Tuple

import requests
import torch
import mido


MAESTRO_VERSION = "v3.0.0"
MAESTRO_BASE = f"https://storage.googleapis.com/magentadata/datasets/maestro/{MAESTRO_VERSION}"
MAESTRO_CSV = f"{MAESTRO_BASE}/maestro-{MAESTRO_VERSION}.csv"
MAESTRO_MIDI_ZIP = f"{MAESTRO_BASE}/maestro-{MAESTRO_VERSION}-midi.zip"


def _download(url: str, path: str) -> None:
    """Download a file to path if it does not exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with open(path, "wb") as f:
        f.write(resp.content)


def _ensure_maestro(data_dir: str) -> Tuple[str, str]:
    """Ensure MAESTRO CSV and MIDI data are available and return paths."""
    cache_dir = os.path.join(data_dir, "cache")
    csv_path = os.path.join(cache_dir, f"maestro-{MAESTRO_VERSION}.csv")
    zip_path = os.path.join(cache_dir, f"maestro-{MAESTRO_VERSION}-midi.zip")
    midi_root = os.path.join(data_dir, f"maestro-{MAESTRO_VERSION}-midi")

    _download(MAESTRO_CSV, csv_path)
    _download(MAESTRO_MIDI_ZIP, zip_path)

    if not os.path.exists(midi_root):
        os.makedirs(midi_root, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(midi_root)
    # Some archives extract into an extra nested folder.
    nested_root = os.path.join(midi_root, f"maestro-{MAESTRO_VERSION}")
    if os.path.isdir(nested_root):
        midi_root = nested_root
    return csv_path, midi_root


def _read_metadata(csv_path: str) -> Dict[str, List[str]]:
    """Read MAESTRO metadata and return split -> list of MIDI filenames."""
    splits = {"train": [], "valid": [], "test": []}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row["split"]
            if split == "validation":
                split = "valid"
            midi_filename = row["midi_filename"]
            if split in splits:
                splits[split].append(midi_filename)
    return splits


def _midi_to_onset_steps(
    midi_path: str,
    steps_per_beat: int = 4,
) -> Dict[int, List[int]]:
    """Convert a MIDI file to onset steps at a fixed temporal resolution."""
    mid = mido.MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat
    step_ticks = max(1, ticks_per_beat // steps_per_beat)
    onsets: Dict[int, List[int]] = {}

    for track in mid.tracks:
        abs_time = 0
        for msg in track:
            abs_time += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                step = int(round(abs_time / step_ticks))
                onsets.setdefault(step, []).append(msg.note)
    return onsets


def _encode_onset_sequence(
    onsets: Dict[int, List[int]],
    note_on_base: int,
    time_shift_id: int,
    bar_id: int,
    phrase_id: int,
    min_midi: int,
    max_midi: int,
    bar_every: int = 16,
    phrase_every_bars: int = 4,
) -> List[int]:
    """Encode onset steps into a note-on/time-shift token sequence."""
    if not onsets:
        return []
    max_step = max(onsets.keys())
    tokens: List[int] = []
    for i in range(max_step + 1):
        if bar_every > 0 and i % bar_every == 0:
            tokens.append(bar_id)
            if phrase_every_bars > 0 and (i // bar_every) % phrase_every_bars == 0:
                tokens.append(phrase_id)
        pitches = sorted(onsets.get(i, []))
        for pitch in pitches:
            pitch = max(min_midi, min(max_midi, pitch))
            tokens.append(note_on_base + (pitch - min_midi))
        tokens.append(time_shift_id)
    return tokens


def load_maestro_piano(
    data_dir: str,
    block_size: int = 512,
    min_midi: int = 21,
    max_midi: int = 108,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Load MAESTRO MIDI-only subset and return split tensors and vocab."""
    csv_path, midi_root = _ensure_maestro(data_dir)
    splits = _read_metadata(csv_path)

    note_on_base = 0
    note_on_count = max_midi - min_midi + 1
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
        "rest_pitch": min_midi - 1,
    }

    # Cache full token streams (independent of block_size)
    cache_dir = os.path.join(data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    token_cache_path = os.path.join(
        cache_dir, f"maestro_{MAESTRO_VERSION}_tokens.pt"
    )
    def build_token_streams() -> Dict[str, torch.Tensor]:
        token_streams: Dict[str, torch.Tensor] = {}
        for split_name, files in splits.items():
            tokens_all: List[int] = []
            missing = 0
            for midi_rel in files:
                midi_path = os.path.join(midi_root, midi_rel)
                if not os.path.exists(midi_path):
                    missing += 1
                    continue
                onsets = _midi_to_onset_steps(midi_path, steps_per_beat=4)
                tokens_all.append(bos_id)
                tokens_all.extend(
                    _encode_onset_sequence(
                        onsets,
                        note_on_base,
                        time_shift_id,
                        bar_id,
                        phrase_id,
                        min_midi,
                        max_midi,
                    )
                )
                tokens_all.append(eos_id)
            if missing > 0 and not tokens_all:
                raise FileNotFoundError(
                    f"MAESTRO MIDI files missing under {midi_root}. "
                    "Please re-download the dataset."
                )
            token_streams[split_name] = torch.tensor(tokens_all, dtype=torch.long)
        return token_streams

    if os.path.exists(token_cache_path):
        token_streams = torch.load(token_cache_path)
        # If cache was built with wrong split names, rebuild it.
        if (
            "valid" not in token_streams
            or token_streams["valid"].numel() == 0
            or token_streams["train"].numel() == 0
        ):
            token_streams = build_token_streams()
            torch.save(token_streams, token_cache_path)
    else:
        token_streams = build_token_streams()
        torch.save(token_streams, token_cache_path)

    def split_to_tensor(split: str) -> torch.Tensor:
        tokens = token_streams[split]
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
