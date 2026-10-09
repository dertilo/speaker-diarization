"""Audio loading helper: decode any container to mono float32 @ 16 kHz."""

from __future__ import annotations

import av
import numpy as np


def load_mono_16k(path: str) -> np.ndarray:
    """Decode an audio file to a mono float32 PCM array at 16 kHz."""
    container = av.open(path)
    stream = container.streams.audio[0]
    resampler = av.audio.resampler.AudioResampler(
        format="flt",
        layout="mono",
        rate=16_000,
    )
    chunks: list[np.ndarray] = []
    for frame in container.decode(stream):
        for resampled in resampler.resample(frame):
            chunks.append(resampled.to_ndarray().reshape(-1))
    tail = resampler.resample(None)
    for resampled in tail:
        chunks.append(resampled.to_ndarray().reshape(-1))
    container.close()
    return np.concatenate(chunks).astype(np.float32)
