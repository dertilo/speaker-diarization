"""Merge per-window cluster labels into non-overlapping speaker turns."""

from __future__ import annotations

from speaker_diarization.rttm import Turn
from speaker_diarization.vad import Span


def windows_to_turns(
    windows: list,
    labels: list[int],
    min_gap: float = 0.5,
) -> list[Turn]:
    """windows: list of objects with .start/.end, in time order. Consecutive
    same-label windows merge into one turn; a following same-label window
    separated by a gap <= min_gap still merges into the same turn.
    Noise windows (label == -1) become turns labelled "noise".
    """
    turns: list[Turn] = []
    for w, label in zip(windows, labels, strict=True):
        name = "noise" if label == -1 else f"speaker_{label}"
        if turns and turns[-1].label == name and w.start - turns[-1].end <= min_gap:
            turns[-1] = Turn(start=turns[-1].start, end=max(turns[-1].end, w.end), label=name)
        else:
            start = w.start
            if turns and w.start < turns[-1].end:
                # Overlapping windows (e.g. 1.5s window / 0.75s hop) at a
                # label change would otherwise make the hyp turns overlap,
                # which the scorer counts as a false alarm. Cut at the
                # midpoint of the overlap instead.
                mid = (w.start + turns[-1].end) / 2
                turns[-1] = Turn(start=turns[-1].start, end=mid, label=turns[-1].label)
                start = mid
            turns.append(Turn(start=start, end=w.end, label=name))
    return turns


def drop_non_speech_clusters(
    turns: list[Turn],
    speech_spans: list[Span],
    min_speech_overlap: float = 0.5,
) -> list[Turn]:
    """Drop every turn belonging to a label whose turns overlap VAD speech
    spans less than min_speech_overlap of their total duration.
    """
    by_label: dict[str, list[Turn]] = {}
    for t in turns:
        by_label.setdefault(t.label, []).append(t)

    def overlap_fraction(label_turns: list[Turn]) -> float:
        total = sum(t.duration for t in label_turns)
        if total <= 0:
            return 0.0
        intersect = sum(
            max(0.0, min(t.end, s.end) - max(t.start, s.start))
            for t in label_turns
            for s in speech_spans
            if t.start < s.end and t.end > s.start
        )
        return intersect / total

    keep_labels = {
        label
        for label, label_turns in by_label.items()
        if overlap_fraction(label_turns) >= min_speech_overlap
    }
    return [t for t in turns if t.label in keep_labels]
