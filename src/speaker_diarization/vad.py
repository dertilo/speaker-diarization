"""Silero VAD via sherpa-onnx: speech spans over a whole mono 16 kHz array."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sherpa_onnx as so


@dataclass(frozen=True)
class Span:
    start: float
    end: float


def detect_speech_spans(
    samples: np.ndarray,
    model_path: str,
    sample_rate: int = 16_000,
    threshold: float = 0.5,
    min_silence_duration: float = 0.5,
    min_speech_duration: float = 0.25,
) -> list[Span]:
    """Run Silero VAD over the full array, chunk-fed in 512-sample windows."""
    vad_config = so.VadModelConfig(
        silero_vad=so.SileroVadModelConfig(
            model=model_path,
            threshold=threshold,
            min_silence_duration=min_silence_duration,
            min_speech_duration=min_speech_duration,
            window_size=512,
        ),
        sample_rate=sample_rate,
    )
    vad = so.VoiceActivityDetector(vad_config, buffer_size_in_seconds=600)

    window = 512
    spans: list[Span] = []
    num_processed = 0
    for i in range(0, len(samples), window):
        chunk = samples[i : i + window]
        if len(chunk) < window:
            chunk = np.pad(chunk, (0, window - len(chunk)))
        vad.accept_waveform(chunk)
        num_processed += window
        while not vad.empty():
            seg = vad.front
            start_s = seg.start / sample_rate
            end_s = start_s + len(seg.samples) / sample_rate
            spans.append(Span(start=start_s, end=end_s))
            vad.pop()
    vad.flush()
    while not vad.empty():
        seg = vad.front
        start_s = seg.start / sample_rate
        end_s = start_s + len(seg.samples) / sample_rate
        spans.append(Span(start=start_s, end=end_s))
        vad.pop()
    return spans


def merge_close_spans(spans: list[Span], max_gap: float = 0.5) -> list[Span]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda s: s.start)
    merged = [spans[0]]
    for s in spans[1:]:
        last = merged[-1]
        if s.start - last.end <= max_gap:
            merged[-1] = Span(start=last.start, end=max(last.end, s.end))
        else:
            merged.append(s)
    return merged
