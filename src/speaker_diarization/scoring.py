"""Score predicted turns against a reference RTTM with pyannote.metrics."""

from __future__ import annotations

from dataclasses import dataclass

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

from speaker_diarization.rttm import Turn


def _to_annotation(turns: list[Turn]) -> Annotation:
    ann = Annotation()
    for t in turns:
        if t.end > t.start:
            ann[Segment(t.start, t.end)] = t.label
    return ann


@dataclass(frozen=True)
class Scores:
    der: float
    confusion: float
    miss: float
    false_alarm: float
    total: float

    def as_percent(self) -> dict[str, float]:
        return {
            "DER": 100 * self.der,
            "SER_confusion": 100 * self.confusion / self.total,
            "miss": 100 * self.miss / self.total,
            "false_alarm": 100 * self.false_alarm / self.total,
        }


def score(
    ref_turns: list[Turn],
    hyp_turns: list[Turn],
    collar: float = 0.25,
) -> Scores:
    metric = DiarizationErrorRate(collar=collar, skip_overlap=True)
    reference = _to_annotation(ref_turns)
    hypothesis = _to_annotation(hyp_turns)
    components = metric(reference, hypothesis, detailed=True)
    der = components["diarization error rate"]
    return Scores(
        der=der,
        confusion=components["confusion"],
        miss=components["missed detection"],
        false_alarm=components["false alarm"],
        total=components["total"],
    )
