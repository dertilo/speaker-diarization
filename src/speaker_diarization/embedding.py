"""TitaNet-large embeddings over fixed-length overlapping windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sherpa_onnx as so


@dataclass(frozen=True)
class EmbeddedWindow:
    start: float
    end: float
    embedding: np.ndarray


def make_windows(
    duration: float,
    window: float = 1.5,
    shift: float = 0.75,
) -> list[tuple[float, float]]:
    """Overlapping (start, end) windows covering [0, duration)."""
    out = []
    start = 0.0
    while start < duration:
        end = min(start + window, duration)
        if end - start >= 0.1:
            out.append((start, end))
        if end >= duration:
            break
        start += shift
    return out


class TitaNetEmbedder:
    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16_000,
        num_threads: int = 1,
    ) -> None:
        """num_threads: intra-op CPU parallelism for a single ONNX Runtime
        session (`SpeakerEmbeddingExtractorConfig.num_threads`). Micro-benchmarked
        against a multiprocessing pool of single-threaded extractors on this
        laptop (i7-8550U, 4 physical / 8 logical cores); intra-op won at every j
        (e.g. j=4: ~26 windows/s vs ~21 windows/s for the pool, plus lower/no
        per-process model-load overhead), so only this path is implemented.
        sherpa-onnx's SpeakerEmbeddingExtractor.compute() takes one OnlineStream
        at a time -- there is no batched-inference entry point to use instead.
        """
        config = so.SpeakerEmbeddingExtractorConfig(
            model=model_path,
            num_threads=num_threads,
        )
        self._extractor = so.SpeakerEmbeddingExtractor(config)
        self._sample_rate = sample_rate

    def embed(self, samples: np.ndarray) -> np.ndarray:
        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate=self._sample_rate, waveform=samples)
        stream.input_finished()
        embedding = self._extractor.compute(stream)
        return np.asarray(embedding, dtype=np.float32)

    def embed_windows(
        self,
        samples: np.ndarray,
        window: float = 1.5,
        shift: float = 0.75,
    ) -> list[EmbeddedWindow]:
        duration = len(samples) / self._sample_rate
        windows = make_windows(duration, window=window, shift=shift)
        out = []
        for start, end in windows:
            seg = samples[round(start * self._sample_rate) : round(end * self._sample_rate)]
            if len(seg) < int(0.1 * self._sample_rate):
                continue
            out.append(EmbeddedWindow(start=start, end=end, embedding=self.embed(seg)))
        return out
