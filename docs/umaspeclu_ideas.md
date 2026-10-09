# umaspeclu: extracted ideas

Read-only study of Tilo's old speaker clusterer, for re-implementation without copying
source. Sources: `nmaudio-speaker-diarization/nmaudio_speaker_diarization/speaker_clusterer.py`,
`diarization/umaspeclu_vad_diarizer.py`, `diarization/umaspeclu_diarizers.py`,
`evaluate_speaker_diarization/evaluate_umaspeclu_diarization.py`, `tests/test_speaker_clusterer.py`,
`README.md` (all under `/home/tilo/code/iais_code/NMAUDIO/nmaudio-speaker-diarization/`).

## Pipeline that produced README's SER 2.08 / DER 17.6 on `oLnl1D6owYA`

1. **VAD**: faster-whisper's Silero VAD (`get_speech_timestamps`), `threshold=0.7`,
   `min_speech_duration_ms=250`, `min_silence_duration_ms=2000`, `speech_pad_ms=400`.
   Adjacent/short speech spans then merged with `max_gap_dur=0.5s`.
   Note: this VAD only gates which embedded windows count as "speech" after clustering
   (see step 5) -- it is not used to pre-segment audio before embedding. Embedding runs
   on fixed-length windows over the *whole* audio.
2. **Windowing for embedding**: overlapping sub-segments, `window=1.5s`, `shift/step=0.75s`
   (NeMo's `get_subsegments` logic), computed per non-overlapping input audio span (here: one span = the whole file).
3. **Embedding**: TitaNet-large (NeMo), one 192-dim vector per window.
4. **Normalization**: NeMo's `embedding_normalize` (mean-center each column across the
   batch, then L2-normalize each row; optional std-scaling is off by default and not
   used here) applied to the full embedding matrix (including calibration anchors, see
   below) before UMAP. Verified against NeMo's source
   (`nemo/collections/asr/parts/utils/speaker_utils.py`, `embedding_normalize`,
   GitHub `NVIDIA/NeMo@main`, 2026-10-09): `embs = embs - embs.mean(axis=0)` then
   `embs = embs / norm(embs, axis=-1)`.
5. **Calibration anchors**: a few manually-labeled reference snippets of the *same* file
   (3 anchors here: 2 known speakers + "DW-jingle" non-speech) are embedded the same way,
   concatenated to the real embeddings before UMAP/HDBSCAN, offset far in time so they
   never collide with real segments, and dropped again after clustering (their pred
   labels are kept only for the "with_calib" diagnostic score). This materially stabilizes
   HDBSCAN/UMAP's cluster shapes on this one file. Treated as a deviation we do **not**
   reproduce (see report) since it uses hand-picked ground truth from the test file itself.
6. **UMAP**: `n_neighbors=60`, `min_dist=0.0`, `n_components=3`, `metric="euclidean"`,
   `random_state=42` (tutorial-recommended defaults were n_neighbors=30/dim=10; the
   evaluation config that hit SER 2.08 used 60/3).
7. **HDBSCAN**: default params except `min_cluster_size=3`; no `min_samples` override.
   Produces per-window integer labels, `-1` = noise.
8. **Windows -> turns**: consecutive windows with the same label are merged into
   non-overlapping speaker turns (first only "raw" merge, then merged again with
   `same_speaker_min_gap_dur=0.5s` to bridge short gaps of the same speaker).
9. **Noise/non-speech filtering**: for each resulting speaker-labeled cluster of turns,
   compute what fraction of its total duration overlaps the VAD speech spans from step 1;
   drop clusters whose speech-overlap fraction is `< 0.5` (filters out e.g. music/jingle
   clusters including HDBSCAN noise if its turns overlap little with VAD speech). Note
   this does **not** drop individual `-1`-labelled windows directly, it is a cluster-level
   filter keyed by majority label.
10. **Scoring**: `speechbrain_DER` (calls `md-eval.pl` with `-r ref_rttm -s sys_rttm`,
    i.e. ref then hyp, standard NIST order -- checked the call site, it is **not**
    swapped), `collar=0.25`, `ignore_overlap=True`. Reports (as percent of reference
    speaker time, the "SCORED SPEAKER TIME" denominator, which depends only on the
    reference, not on how much the hypothesis covers): `miss_speaker`, `fa_speaker`,
    `SER` (= speaker confusion / error), `DER` = sum of the three. The three component
    terms are **not** symmetric in ref/hyp like a naive score -- DER is computed over
    the reference's scored speaker time, which is why the README calls out that the
    hand labels are coarse: predicted turns are finer-grained than the manually drawn
    reference turns, so a lot of correctly-labelled speech falls in gaps between
    reference turns and counts as neither hit nor miss, while genuinely unlabeled
    predicted speech outside any reference turn counts as **false alarm**, not miss
    (miss is reference speaker-time that the system fails to cover at all; confirmed by
    running `md-eval.pl` directly on our own re-implementation's output -- see report
    for the validation round -- its per-run false-alarm time moves with how much extra,
    unmatched speech our predictions contain, while miss stays essentially flat). (README:
    SER 2.08, DER 17.6, of which miss 15.1, fa 0.38 -- the old run's very low FA and high
    miss stem from its coarse/sparse hand-labeled reference plus the calibration-anchor
    assisted clustering dropping most non-speech/stray clusters, not from any ref/hyp
    swap.)

## Params not used for the README number but seen elsewhere

- `speaker_clusterer.py` defaults: `umap_n_neighbors=30`, `umap_dimension=10`,
  `metric="euclidean"`.
- `test_speaker_clusterer.py` uses `metric="cosine"`, own word-alignment-based
  segmentation (not VAD), no VAD-based non-speech filtering, `min_gap_dur=4.0` merge,
  asserts SER < 2.5/3.0 and DER < 5.0 -- a looser, VAD-free sanity check, not the
  README's headline number.
- Scoring collar of 0.25s is NIST convention (0.25s buffer on each side of a reference
  turn boundary, 0.5s total per boundary, errors inside the collar are not scored).

## What we will NOT reproduce

- Calibration anchors seeded from the test file's own ground truth (step 5) -- using
  ground truth of the scored file to help clustering the same file is circular evaluation
  if reproduced naively on this single clip; the plan instead targets the plain
  embed -> UMAP -> HDBSCAN path and reports what SER it reaches without that help.
- NeMo's own VAD/embedding normalization utilities and NeMo/PyTorch entirely (ONNX
  TitaNet-large via sherpa-onnx instead); Silero VAD via sherpa-onnx instead of
  faster-whisper's bundled Silero VAD (same underlying model family, different wrapper).
