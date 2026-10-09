# speaker-diarization

Experiments for telling speakers apart in recordings.

- method: umaspeclu, a re-implementation of an older idea: speaker embeddings per speech window, UMAP reduction, HDBSCAN clustering.
- embedder: TitaNet-large as ONNX, run through sherpa-onnx. No NeMo, no PyTorch.
- metric: speaker confusion (SER) first. DER depends on how coarsely the reference was labeled.

Audio, models and run outputs stay out of git (`data/`, `models/`, `runs/`).
