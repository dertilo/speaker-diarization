"""Normalize -> UMAP -> HDBSCAN clustering of window embeddings."""

from __future__ import annotations

import numpy as np
import umap
from sklearn.cluster import HDBSCAN


def embedding_normalize(embeddings: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Mean-center each column, then L2-normalize each row (NeMo-style)."""
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return centered / np.maximum(norms, eps)


def cluster_embeddings(
    embeddings: np.ndarray,
    n_neighbors: int = 60,
    n_components: int = 3,
    metric: str = "euclidean",
    min_cluster_size: int = 3,
    random_state: int = 42,
) -> np.ndarray:
    """Return an integer cluster label per row, -1 for HDBSCAN noise."""
    normalized = embedding_normalize(embeddings)
    reduced = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.0,
        n_components=n_components,
        metric=metric,
        random_state=random_state,
    ).fit_transform(normalized)
    labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(reduced)
    return labels
