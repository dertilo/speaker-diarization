"""Compare TitaNet-large vs WeSpeaker ResNet34-LM on the DW clip, measuring
seed-to-seed noise rather than tuning: embeddings are computed once per
embedder (deterministic, j=1, cached to runs/), then clustered with UMAP
random_state 0..9 (rest of the pipeline -- normalize, HDBSCAN, turn merging,
VAD filter -- held fixed, no calibration anchors). Each run is scored with
both md-eval.pl (primary) and pyannote.metrics.

Usage: uv run python scripts/compare_embedders.py
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

import numpy as np

from speaker_diarization.audio import load_mono_16k
from speaker_diarization.clustering import cluster_embeddings
from speaker_diarization.embedding import EmbeddedWindow, TitaNetEmbedder
from speaker_diarization.rttm import read_rttm, write_rttm
from speaker_diarization.run_umaspeclu import EMBEDDER_MODELS
from speaker_diarization.scoring import score
from speaker_diarization.turns import drop_non_speech_clusters, windows_to_turns
from speaker_diarization.vad import detect_speech_spans, merge_close_spans

AUDIO = "data/oLnl1D6owYA.opus"
REF_RTTM = "data/oLnl1D6owYA_ref.rttm"
VAD_MODEL = "models/silero_vad.onnx"
RUNS_DIR = Path("runs")
MDEVAL = (
    "/home/tilo/code/iais_code/NMAUDIO/nmaudio-speaker-diarization/"
    "evaluate_speaker_diarization/md-eval.pl"
)
SEEDS = list(range(10))

MISS_RE = re.compile(r"(?<=MISSED SPEAKER TIME =)[\d.]+")
FA_RE = re.compile(r"(?<=FALARM SPEAKER TIME =)[\d.]+")
SER_RE = re.compile(r"(?<=SPEAKER ERROR TIME =)[\d.]+")
SCORED_RE = re.compile(r"(?<=SCORED SPEAKER TIME =)[\d.]+")


def md_eval(ref_rttm: str, sys_rttm: str) -> dict[str, float]:
    cmd = ["perl", MDEVAL, "-af", "-r", ref_rttm, "-s", sys_rttm, "-c", "0.25", "-1"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
    scored = float(SCORED_RE.findall(out)[-1])
    miss = float(MISS_RE.findall(out)[-1])
    fa = float(FA_RE.findall(out)[-1])
    ser = float(SER_RE.findall(out)[-1])
    return {
        "SER": 100 * ser / scored,
        "miss": 100 * miss / scored,
        "FA": 100 * fa / scored,
        "DER": 100 * (ser + miss + fa) / scored,
    }


def embed_once(embedder_name: str, samples: np.ndarray, jobs: int) -> tuple[list[EmbeddedWindow], float]:
    model_path = EMBEDDER_MODELS[embedder_name]
    t0 = time.time()
    embedder = TitaNetEmbedder(model_path, num_threads=jobs)
    windows = embedder.embed_windows(samples, window=1.5, shift=0.75)
    dt = time.time() - t0
    return windows, dt


def cache_path(embedder_name: str) -> Path:
    return RUNS_DIR / f"{embedder_name}_embeddings_j1.npz"


def load_or_embed(embedder_name: str, samples: np.ndarray) -> tuple[list[EmbeddedWindow], float]:
    path = cache_path(embedder_name)
    if path.exists():
        data = np.load(path)
        windows = [
            EmbeddedWindow(start=s, end=e, embedding=emb)
            for s, e, emb in zip(data["starts"], data["ends"], data["embeddings"], strict=True)
        ]
        return windows, float(data["embed_seconds"])
    windows, dt = embed_once(embedder_name, samples, jobs=1)
    RUNS_DIR.mkdir(exist_ok=True)
    np.savez(
        path,
        starts=np.array([w.start for w in windows]),
        ends=np.array([w.end for w in windows]),
        embeddings=np.stack([w.embedding for w in windows]),
        embed_seconds=dt,
    )
    return windows, dt


def one_seed_run(
    embedder_name: str,
    windows: list[EmbeddedWindow],
    speech_spans,
    ref_turns,
    seed: int,
) -> dict[str, float]:
    embeddings = np.stack([w.embedding for w in windows])
    labels = cluster_embeddings(
        embeddings,
        n_neighbors=60,
        n_components=3,
        metric="euclidean",
        min_cluster_size=3,
        random_state=seed,
    )
    turns = windows_to_turns(windows, labels.tolist(), min_gap=0.5)
    turns = drop_non_speech_clusters(turns, speech_spans, min_speech_overlap=0.5)

    rttm_path = RUNS_DIR / f"{embedder_name}_seed{seed}.rttm"
    write_rttm(str(rttm_path), "oLnl1D6owYA", turns)

    pyannote_scores = score(ref_turns, turns, collar=0.25).as_percent()
    mdeval_scores = md_eval(REF_RTTM, str(rttm_path))
    num_speakers = len({t.label for t in turns})
    return {
        "pyannote": pyannote_scores,
        "mdeval": mdeval_scores,
        "num_speakers": num_speakers,
    }


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.array(values)
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "std": float(arr.std()),
    }


def main() -> None:
    samples = load_mono_16k(AUDIO)
    ref_turns = read_rttm(REF_RTTM)
    num_ref_speakers = len({t.label for t in ref_turns})

    t0 = time.time()
    speech_spans = detect_speech_spans(samples, VAD_MODEL, threshold=0.7)
    speech_spans = merge_close_spans(speech_spans, max_gap=0.5)
    print(f"VAD: {len(speech_spans)} spans ({time.time() - t0:.1f}s)")

    results: dict[str, list[dict]] = {}
    embed_timings_j1: dict[str, float] = {}
    for embedder_name in EMBEDDER_MODELS:
        windows, dt_j1 = load_or_embed(embedder_name, samples)
        embed_timings_j1[embedder_name] = dt_j1
        print(f"{embedder_name}: {len(windows)} windows, embed(j=1) {dt_j1:.1f}s")

        runs = []
        t_sweep0 = time.time()
        for seed in SEEDS:
            runs.append(one_seed_run(embedder_name, windows, speech_spans, ref_turns, seed))
        sweep_dt = time.time() - t_sweep0
        print(f"{embedder_name}: seed sweep took {sweep_dt:.1f}s for {len(SEEDS)} seeds "
              f"({sweep_dt / len(SEEDS):.2f}s/seed, incl. first-call numba compile)")
        results[embedder_name] = runs

    # j=4 embedding timing only, no caching/scoring of this run.
    embed_timings_j4: dict[str, float] = {}
    for embedder_name in EMBEDDER_MODELS:
        _windows, dt_j4 = embed_once(embedder_name, samples, jobs=4)
        embed_timings_j4[embedder_name] = dt_j4

    print("\n=== Summary (md-eval.pl, 10 UMAP seeds, no calibration anchors) ===")
    for embedder_name, runs in results.items():
        ser_vals = [r["mdeval"]["SER"] for r in runs]
        der_vals = [r["mdeval"]["DER"] for r in runs]
        nspk_vals = [r["num_speakers"] for r in runs]
        print(f"\n-- {embedder_name} ({EMBEDDER_MODELS[embedder_name]}) --")
        print(f"  SER (md-eval): {summarize(ser_vals)}")
        print(f"  DER (md-eval): {summarize(der_vals)}")
        print(f"  num_speakers: mean={np.mean(nspk_vals):.1f} "
              f"min={min(nspk_vals)} max={max(nspk_vals)} (ref: {num_ref_speakers})")
        ser_pyannote = [r["pyannote"]["SER_confusion"] for r in runs]
        der_pyannote = [r["pyannote"]["DER"] for r in runs]
        print(f"  SER (pyannote): {summarize(ser_pyannote)}")
        print(f"  DER (pyannote): {summarize(der_pyannote)}")
        print(f"  embed wall time: j=1 {embed_timings_j1[embedder_name]:.1f}s, "
              f"j=4 {embed_timings_j4[embedder_name]:.1f}s")


if __name__ == "__main__":
    main()
