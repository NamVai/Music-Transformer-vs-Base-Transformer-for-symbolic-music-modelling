from typing import List

try:
    import mido
except Exception as exc:
    mido = None
    _MIDO_ERR = exc


def tokens_to_midi(
    tokens: List[int],
    output_path: str,
    note_on_base: int,
    time_shift_id: int,
    min_midi: int = 21,
    rest_pitch: int = 20,
    ticks_per_step: int = 120,
) -> None:
    """Convert event tokens to a simple single-track MIDI file."""
    if mido is None:
        raise RuntimeError(
            "MIDI export requires 'mido'. Install it with: pip install mido"
        ) from _MIDO_ERR
    # Simple event-based conversion:
    # NOTE_ON tokens emit a note with fixed duration (one time step).
    # TIME_SHIFT tokens advance time by ticks_per_step.
    # Type 0 (single track) is the most compatible with simple players.
    mid = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    track.append(mido.Message("program_change", program=0, time=0))

    pending_time = 0
    for tok in tokens:
        if tok == time_shift_id:
            pending_time += ticks_per_step
            continue
        if note_on_base <= tok < time_shift_id:
            pitch = (tok - note_on_base) + rest_pitch
            if pitch == rest_pitch:
                continue
            track.append(
                mido.Message("note_on", note=pitch, velocity=64, time=pending_time)
            )
            # Fixed duration: one step
            track.append(
                mido.Message("note_off", note=pitch, velocity=0, time=ticks_per_step)
            )
            pending_time = 0

    mid.save(output_path)
