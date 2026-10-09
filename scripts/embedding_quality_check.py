"""Clustering-free embedding quality check, per embedder, on cached
embeddings from scripts/compare_embedders.py (runs/{embedder}_embeddings_j1.npz).

Restricts to windows that lie fully inside a single reference turn (no
overlap with any other reference speaker's turn; a window may still touch
a silence gap, collar not needed). For those windows reports:
  - count of windows and distinct reference speakers used
  - mean cosine similarity same-speaker vs different-speaker (over all pairs)
  - EER of the same/different-speaker binary trial task (cosine score)
  - leave-one-out 1-NN speaker accuracy, excluding neighbours from the same
    reference turn (so adjacent, near-identical windows can't trivially win)

Run twice: raw embeddings (cosine) and after embedding_normalize
(mean-center + L2, i.e. what the pipeline actually clusters on).

Usage: uv run python scripts/embedding_quality_check.py
"""

from __future__ import annotations

import numpy as np

from speaker_diarization.clustering import embedding_normalize
from speaker_diarization.rttm import read_rttm

REF_RTTM = "data/oLnl1D6owYA_ref.rttm"
EMBEDDERS = ["titanet", "resnet34"]


def windows_fully_inside_single_ref_turn(
    starts: np.ndarray,
    ends: np.ndarray,
    ref_turns: list,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Return (mask, speaker_per_kept_window, ref_turn_index_per_kept_window)."""
    speakers = []
    turn_idx = []
    mask = np.zeros(len(starts), dtype=bool)
    for i, (s, e) in enumerate(zip(starts, ends, strict=True)):
        # all ref turns overlapping this window
        overlapping = [
            (ti, t) for ti, t in enumerate(ref_turns) if t.start < e and t.end > s
        ]
        if len(overlapping) != 1:
            continue
        ti, t = overlapping[0]
        if s < t.start or e > t.end:
            continue  # not fully inside
        mask[i] = True
        speakers.append(t.label)
        turn_idx.append(ti)
    return mask, speakers, turn_idx


def cosine_sim_matrix(x: np.ndarray) -> np.ndarray:
    n = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-10)
    return n @ n.T


def eer(same_scores: np.ndarray, diff_scores: np.ndarray) -> float:
    scores = np.concatenate([same_scores, diff_scores])
    labels = np.concatenate([np.ones_like(same_scores), np.zeros_like(diff_scores)])
    order = np.argsort(-scores)
    scores, labels = scores[order], labels[order]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    fnr = 1 - tp / n_pos
    fpr = fp / n_neg
    diffs = np.abs(fnr - fpr)
    i = np.argmin(diffs)
    return float((fnr[i] + fpr[i]) / 2)


def one_nn_accuracy_loo(
    sims: np.ndarray,
    speakers: list[str],
    turn_idx: list[int],
) -> float:
    n = len(speakers)
    correct = 0
    for i in range(n):
        row = sims[i].copy()
        row[i] = -np.inf
        for j in range(n):
            if turn_idx[j] == turn_idx[i]:
                row[j] = -np.inf
        j = int(np.argmax(row))
        if speakers[j] == speakers[i]:
            correct += 1
    return correct / n


def report(name: str, embeddings: np.ndarray, speakers: list[str], turn_idx: list[int]) -> None:
    sims = cosine_sim_matrix(embeddings)
    n = len(speakers)
    same_scores, diff_scores = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if speakers[i] == speakers[j]:
                same_scores.append(sims[i, j])
            else:
                diff_scores.append(sims[i, j])
    same_scores = np.array(same_scores)
    diff_scores = np.array(diff_scores)
    acc = one_nn_accuracy_loo(sims, speakers, turn_idx)
    e = eer(same_scores, diff_scores)
    print(f"  [{name}] windows={n} speakers={len(set(speakers))}")
    print(f"    mean cos same-speaker={same_scores.mean():.4f}  diff-speaker={diff_scores.mean():.4f}")
    print(f"    EER={100 * e:.2f}%  1-NN-LOO-acc={100 * acc:.2f}%")


def main() -> None:
    ref_turns = read_rttm(REF_RTTM)
    for embedder in EMBEDDERS:
        path = f"runs/{embedder}_embeddings_j1.npz"
        d = np.load(path)
        starts, ends, embeddings = d["starts"], d["ends"], d["embeddings"]
        mask, speakers, turn_idx = windows_fully_inside_single_ref_turn(starts, ends, ref_turns)
        emb = embeddings[mask]
        print(f"{embedder}: {mask.sum()}/{len(mask)} windows fully inside a single ref turn")
        report("raw", emb, speakers, turn_idx)
        report("normalized (mean-center+L2)", embedding_normalize(emb), speakers, turn_idx)
        print()


if __name__ == "__main__":
    main()
