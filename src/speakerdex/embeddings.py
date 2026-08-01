"""Speaker-embedding backends.

A backend is anything implementing EmbeddingBackend: it turns a mono float32
waveform at 16 kHz into a single L2-normalized vector. Heavy dependencies are
imported lazily so the core package stays light and testable.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingBackend(Protocol):
    name: str

    def embed(self, wave: np.ndarray, sr: int) -> np.ndarray:
        """Return one L2-normalized 1-D embedding for the given waveform."""
        ...


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def combine(embeddings: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Duration-weighted mean of chunk embeddings, re-normalized."""
    if not embeddings:
        raise ValueError("no embeddings to combine")
    mat = np.stack([l2_normalize(e) for e in embeddings])
    if weights is None:
        mean = mat.mean(axis=0)
    else:
        w = np.asarray(weights, dtype=np.float32)
        mean = (mat * (w / w.sum())[:, None]).sum(axis=0)
    return l2_normalize(mean)


class EcapaBackend:
    """ECAPA-TDNN speaker embeddings via SpeechBrain (speechbrain/spkrec-ecapa-voxceleb).

    Requires the 'ecapa' extra:  pip install "speakerdex[ecapa]"
    """

    name = "ecapa"

    def __init__(self, device: str | None = None):
        try:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as e:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "The ECAPA backend needs the optional dependencies. "
                'Install them with: pip install "speakerdex[ecapa]"'
            ) from e
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._torch = torch
        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device},
        )

    def embed(self, wave: np.ndarray, sr: int) -> np.ndarray:  # pragma: no cover
        if sr != 16_000:
            raise ValueError(f"EcapaBackend expects 16 kHz audio, got {sr}")
        tensor = self._torch.from_numpy(np.ascontiguousarray(wave)).unsqueeze(0)
        with self._torch.no_grad():
            emb = self._model.encode_batch(tensor)
        return l2_normalize(emb.squeeze().cpu().numpy())


_BACKENDS = {"ecapa": EcapaBackend}


def get_backend(name: str, **kwargs) -> EmbeddingBackend:
    try:
        cls = _BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown backend {name!r}; available: {sorted(_BACKENDS)}") from None
    return cls(**kwargs)


def register_backend(name: str, cls: type) -> None:
    """Register a custom backend class (used by tests and plugins)."""
    _BACKENDS[name] = cls
