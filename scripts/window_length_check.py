"""Addendum 5c: repeat the clustering-free embedding-quality check (EER,
1-NN-LOO accuracy) for both embedders at window lengths 1.5s (reuse cached
embeddings), 3s and 6s (shift = window/2), restricted to the first 200s of
the DW clip and capped at 300 windows per (embedder, window length), to see
whether ResNet34 recovers at longer windows.
"""

from __future__ import annotations

import numpy as np

from speaker_diarization.audio import load_mono_16k
from speaker_diarization.clustering import embedding_normalize
from speaker_diarization.embedding import TitaNetEmbedder
from speaker_diarization.rttm import read_rttm
from speaker_diarization.run_umaspeclu import EMBEDDER_MODELS
from scripts.embedding_quality_check import (
    windows_fully_inside_single_ref_turn,
    cosine_sim_matrix,
    eer,
    one_nn_accuracy_loo,
)

AUDIO = "data/oLnl1D6owYA.opus"
REF_RTTM = "data/oLnl1D6owYA_ref.rttm"
SUBSET_SECONDS = 200
CAP = 300
WINDOW_LENGTHS = [1.5, 3.0, 6.0]


def eval_one(embedder_name, window_s, starts, ends, embeddings, ref_turns):
    mask, speakers, turn_idx = windows_fully_inside_single_ref_turn(starts, ends, ref_turns)
    emb = embeddings[mask][:CAP]
    speakers = speakers[:CAP]
    turn_idx = turn_idx[:CAP]
    n = len(speakers)
    if n < 4 or len(set(speakers)) < 2:
        print(f"  [{embedder_name} w={window_s}s] too few qualifying windows ({n}), skipping")
        return
    normed = embedding_normalize(emb)
    for tag, data in [("raw", emb), ("normalized", normed)]:
        sims = cosine_sim_matrix(data)
        same, diff = [], []
        for i in range(n):
            for j in range(i + 1, n):
                (same if speakers[i] == speakers[j] else diff).append(sims[i, j])
        same, diff = np.array(same), np.array(diff)
        acc = one_nn_accuracy_loo(sims, speakers, turn_idx)
        e = eer(same, diff) if len(same) and len(diff) else float("nan")
        print(f"  [{embedder_name} w={window_s}s {tag}] n={n} spk={len(set(speakers))} "
              f"EER={100*e:.2f}% 1-NN-acc={100*acc:.2f}%")


def main() -> None:
    samples = load_mono_16k(AUDIO)
    subset = samples[: int(16000 * SUBSET_SECONDS)]
    ref_turns = [t for t in read_rttm(REF_RTTM) if t.start < SUBSET_SECONDS]

    for embedder_name, path in EMBEDDER_MODELS.items():
        embedder = TitaNetEmbedder(path, num_threads=1)
        for w in WINDOW_LENGTHS:
            if w == 1.5:
                d = np.load(f"runs/{embedder_name}_embeddings_j1.npz")
                keep = d["ends"] <= SUBSET_SECONDS
                starts, ends, embeddings = d["starts"][keep], d["ends"][keep], d["embeddings"][keep]
            else:
                windows = embedder.embed_windows(subset, window=w, shift=w / 2)
                starts = np.array([x.start for x in windows])
                ends = np.array([x.end for x in windows])
                embeddings = np.stack([x.embedding for x in windows])
            eval_one(embedder_name, w, starts, ends, embeddings, ref_turns)


if __name__ == "__main__":
    main()
