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


# Exact calibration-anchor spans from the old eval config
# (nmaudio_speaker_diarization/evaluate_speaker_diarization/evaluate_umaspeclu_diarization.py,
# `umaspeclu_diarizers`, `CalibrationDatum.start_end_labels`), used only validation,
# opt-in via --use-calibration-anchors.
CALIBRATION_ANCHORS = [
    (0.662, 0.662 + 9.082, "eddy_micah_jr"),
    (184.167, 184.167 + 12.384, "women_in_street_blue_jacket"),
    (75.0, 85.0, "DW-jingle"),
]


def run(
    audio_path: str,
    ref_rttm_path: str,
    titanet_model: str,
    vad_model: str,
    out_rttm_path: str | None = None,
    use_calibration_anchors: bool = False,
) -> None:
    t0 = time.time()
    samples = load_mono_16k(audio_path)
    duration = len(samples) / 16_000
    print(f"loaded audio: {duration:.1f}s")

    embedder = TitaNetEmbedder(titanet_model)
    windows = embedder.embed_windows(samples, window=1.5, shift=0.75)
    print(f"embedded {len(windows)} windows ({time.time() - t0:.1f}s so far)")

    import numpy as np

    num_anchors = 0
    anchor_embeddings = None
    if use_calibration_anchors:
        sr = 16_000
        anchor_embeddings = np.stack(
            [
                embedder.embed(samples[round(s * sr) : round(e * sr)])
                for s, e, _label in CALIBRATION_ANCHORS
            ],
        )
        num_anchors = len(CALIBRATION_ANCHORS)
        print(f"embedded {num_anchors} calibration anchors")

    embeddings = np.stack([w.embedding for w in windows])
    if use_calibration_anchors:
        embeddings = np.concatenate([embeddings, anchor_embeddings], axis=0)

    labels = cluster_embeddings(
        embeddings,
        n_neighbors=60,
        n_components=3,
        metric="euclidean",
        min_cluster_size=3,
    )
    print(f"clusters found: {sorted(set(labels.tolist()))}")

    if use_calibration_anchors:
        labels = labels[: -num_anchors]  # drop anchor labels before turn-building

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
    return runtime, scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="data/oLnl1D6owYA.opus")
    parser.add_argument("--ref-rttm", default="data/oLnl1D6owYA_ref.rttm")
    parser.add_argument("--titanet-model", default="models/nemo_en_titanet_large.onnx")
    parser.add_argument("--vad-model", default="models/silero_vad.onnx")
    parser.add_argument("--out-rttm", default="runs/oLnl1D6owYA_pred.rttm")
    parser.add_argument(
        "--use-calibration-anchors",
        action="store_true",
        help=(
            "Opt-in: seed clustering with the 3 calibration anchors from the old "
            "eval config (exact spans of this same file). For validation against "
            "the old pipeline only, not a fair unsupervised score."
        ),
    )
    args = parser.parse_args()
    run(
        args.audio,
        args.ref_rttm,
        args.titanet_model,
        args.vad_model,
        args.out_rttm,
        use_calibration_anchors=args.use_calibration_anchors,
    )


if __name__ == "__main__":
    main()
