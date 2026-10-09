"""Reproduce umaspeclu (TitaNet -> UMAP -> HDBSCAN) on one clip and score it."""

from __future__ import annotations

import argparse
import time

from speaker_diarization.audio import load_mono_16k
from speaker_diarization.clustering import cluster_embeddings
from speaker_diarization.embedding import TitaNetEmbedder
from speaker_diarization.rttm import read_rttm, write_rttm
from speaker_diarization.scoring import score
from speaker_diarization.turns import drop_non_speech_clusters, windows_to_turns
from speaker_diarization.vad import detect_speech_spans, merge_close_spans


def run(
    audio_path: str,
    ref_rttm_path: str,
    titanet_model: str,
    vad_model: str,
    out_rttm_path: str | None = None,
) -> None:
    t0 = time.time()
    samples = load_mono_16k(audio_path)
    duration = len(samples) / 16_000
    print(f"loaded audio: {duration:.1f}s")

    embedder = TitaNetEmbedder(titanet_model)
    windows = embedder.embed_windows(samples, window=1.5, shift=0.75)
    print(f"embedded {len(windows)} windows ({time.time() - t0:.1f}s so far)")

    import numpy as np

    embeddings = np.stack([w.embedding for w in windows])
    labels = cluster_embeddings(
        embeddings,
        n_neighbors=60,
        n_components=3,
        metric="euclidean",
        min_cluster_size=3,
    )
    print(f"clusters found: {sorted(set(labels.tolist()))}")

    turns = windows_to_turns(windows, labels.tolist(), min_gap=0.5)

    speech_spans = detect_speech_spans(samples, vad_model, threshold=0.7)
    speech_spans = merge_close_spans(speech_spans, max_gap=0.5)
    print(f"VAD speech spans: {len(speech_spans)}")

    turns = drop_non_speech_clusters(turns, speech_spans, min_speech_overlap=0.5)
    print(f"turns after non-speech filtering: {len(turns)}")

    runtime = time.time() - t0
    print(f"total runtime: {runtime:.1f}s ({duration / runtime:.1f}x realtime)")

    if out_rttm_path:
        write_rttm(out_rttm_path, "oLnl1D6owYA", turns)

    ref_turns = read_rttm(ref_rttm_path)
    scores = score(ref_turns, turns, collar=0.25)
    print(scores.as_percent())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="data/oLnl1D6owYA.opus")
    parser.add_argument("--ref-rttm", default="data/oLnl1D6owYA_ref.rttm")
    parser.add_argument("--titanet-model", default="models/nemo_en_titanet_large.onnx")
    parser.add_argument("--vad-model", default="models/silero_vad.onnx")
    parser.add_argument("--out-rttm", default="runs/oLnl1D6owYA_pred.rttm")
    args = parser.parse_args()
    run(args.audio, args.ref_rttm, args.titanet_model, args.vad_model, args.out_rttm)


if __name__ == "__main__":
    main()
