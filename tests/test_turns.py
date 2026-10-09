"""Unit tests for windows_to_turns, in particular the overlap-at-label-change fix."""

from __future__ import annotations

from dataclasses import dataclass

from speaker_diarization.turns import windows_to_turns


@dataclass
class Window:
    start: float
    end: float


def make_windows(n: int, window: float = 1.5, shift: float = 0.75) -> list[Window]:
    return [Window(start=i * shift, end=i * shift + window) for i in range(n)]


def test_no_overlaps_in_output() -> None:
    windows = make_windows(10)
    labels = [0, 0, 0, 1, 1, 1, 2, 2, 0, 0]
    turns = windows_to_turns(windows, labels)
    for a, b in zip(turns, turns[1:]):
        assert a.end <= b.start, f"turns overlap: {a} and {b}"


def test_same_label_merging() -> None:
    windows = make_windows(5)
    labels = [0, 0, 0, 0, 0]
    turns = windows_to_turns(windows, labels)
    assert len(turns) == 1
    assert turns[0].start == windows[0].start
    assert turns[0].end == windows[-1].end


def test_label_change_in_middle_cuts_at_overlap_midpoint() -> None:
    # windows: 1.5s long, 0.75s hop -> window i covers [i*0.75, i*0.75+1.5]
    windows = make_windows(4)
    labels = [0, 0, 1, 1]
    turns = windows_to_turns(windows, labels)
    assert len(turns) == 2
    # window[1] ends at 0.75+1.5=2.25; window[2] starts at 1.5.
    # overlap is [1.5, 2.25], midpoint 1.875.
    midpoint = (windows[2].start + windows[1].end) / 2
    assert turns[0].end == midpoint
    assert turns[1].start == midpoint
    assert turns[0].label == "speaker_0"
    assert turns[1].label == "speaker_1"


def test_noise_label_becomes_noise_turn() -> None:
    windows = make_windows(3)
    labels = [-1, -1, 0]
    turns = windows_to_turns(windows, labels)
    assert turns[0].label == "noise"
    assert turns[-1].label == "speaker_0"
    for a, b in zip(turns, turns[1:]):
        assert a.end <= b.start
