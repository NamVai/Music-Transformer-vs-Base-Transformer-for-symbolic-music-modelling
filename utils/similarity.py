from typing import List

import numpy as np


def token_self_similarity(tokens: List[int], max_len: int = 2048) -> np.ndarray:
    """Compute a simple token equality self-similarity matrix."""
    seq = np.array(tokens[:max_len], dtype=np.int64)
    if seq.size == 0:
        return np.zeros((1, 1), dtype=np.float32)
    sim = (seq[:, None] == seq[None, :]).astype(np.float32)
    return sim
