"""Minimal RTTM read/write for speaker turns: (start, end, label) tuples."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    label: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def read_rttm(path: str) -> list[Turn]:
    turns = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            dur = float(parts[4])
            label = parts[7]
            turns.append(Turn(start=start, end=start + dur, label=label))
    return sorted(turns, key=lambda t: t.start)


def write_rttm(path: str, file_id: str, turns: list[Turn]) -> None:
    with open(path, "w") as f:
        for t in turns:
            f.write(
                f"SPEAKER {file_id} 1 {t.start:.3f} {t.duration:.3f} "
                f"<NA> <NA> {t.label} <NA> <NA>\n",
            )
