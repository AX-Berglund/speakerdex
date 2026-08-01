"""Audio loading and per-cluster segment extraction.

Everything is normalized to mono float32 at TARGET_SR before embedding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from .types import Segment

TARGET_SR = 16_000

# Segment selection defaults (tunable via pipeline config)
MIN_SEGMENT_SEC = 0.5  # ignore blips shorter than this
MAX_CHUNK_SEC = 10.0  # embed in windows of at most this length
MAX_TOTAL_SEC = 60.0  # stop once we have this much speech for a cluster


def load_audio(path: str | Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32 at target_sr."""
    wave, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)  # mixdown to mono
    if sr != target_sr:
        wave = soxr.resample(wave, sr, target_sr)
        sr = target_sr
    return np.ascontiguousarray(wave, dtype=np.float32), sr


def slice_segment(wave: np.ndarray, sr: int, seg: Segment) -> np.ndarray:
    a = max(0, int(seg.start * sr))
    b = min(len(wave), int(seg.end * sr))
    return wave[a:b]


def select_chunks(
    wave: np.ndarray,
    sr: int,
    segments: list[Segment],
    min_segment_sec: float = MIN_SEGMENT_SEC,
    max_chunk_sec: float = MAX_CHUNK_SEC,
    max_total_sec: float = MAX_TOTAL_SEC,
) -> list[np.ndarray]:
    """Pick audio chunks for one cluster, longest segments first.

    Long segments are split into windows of max_chunk_sec. Collection stops
    once max_total_sec of speech has been gathered — enough for a stable
    embedding without paying to embed a whole episode.
    """
    chunks: list[np.ndarray] = []
    total = 0.0
    for seg in sorted(segments, key=lambda s: s.duration, reverse=True):
        if seg.duration < min_segment_sec:
            continue
        audio = slice_segment(wave, sr, seg)
        step = int(max_chunk_sec * sr)
        for i in range(0, len(audio), step):
            chunk = audio[i : i + step]
            if len(chunk) < int(min_segment_sec * sr):
                continue
            chunks.append(chunk)
            total += len(chunk) / sr
            if total >= max_total_sec:
                return chunks
    return chunks
