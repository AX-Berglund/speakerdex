"""Shared test fixtures: synthetic 'voices' and a model-free spectral backend.

Real speaker-embedding models are far too heavy for CI. Instead, each fake
'voice' is a harmonic tone with a distinct fundamental frequency, and the fake
backend embeds audio by its spectral band profile — so the same voice yields
similar embeddings across files, and different voices don't. That is exactly
the property speakerdex relies on, without any ML dependencies.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from speakerdex.embeddings import l2_normalize, register_backend

SR = 16_000


def synth_voice(f0: float, seconds: float, seed: int = 0) -> np.ndarray:
    """A crude 'voice': harmonic stack at fundamental f0 with a little noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    wave = np.zeros_like(t)
    for k in range(1, 4):
        wave += (0.6 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi))
    wave += 0.01 * rng.standard_normal(len(t))
    return (wave / np.abs(wave).max() * 0.9).astype(np.float32)


def build_track(parts: list[tuple[float | None, float]], seed: int = 0) -> np.ndarray:
    """Concatenate (f0, seconds) parts into one track; f0=None means silence."""
    pieces = []
    for i, (f0, seconds) in enumerate(parts):
        if f0 is None:
            pieces.append(np.zeros(int(seconds * SR), dtype=np.float32))
        else:
            pieces.append(synth_voice(f0, seconds, seed=seed + i))
    return np.concatenate(pieces)


class SpectralBackend:
    """Model-free embedding backend: 64 spectral bands up to 4 kHz."""

    name = "fake-spectral"

    def embed(self, wave: np.ndarray, sr: int) -> np.ndarray:
        spec = np.abs(np.fft.rfft(wave))
        freqs = np.fft.rfftfreq(len(wave), 1 / sr)
        mask = freqs < 4000
        spec, freqs = spec[mask], freqs[mask]
        bands = np.zeros(64, dtype=np.float64)
        idx = np.minimum((freqs / 4000 * 64).astype(int), 63)
        np.add.at(bands, idx, spec)
        return l2_normalize(np.sqrt(bands))


register_backend("fake-spectral", SpectralBackend)


@pytest.fixture()
def backend() -> SpectralBackend:
    return SpectralBackend()


@pytest.fixture()
def write_wav(tmp_path):
    def _write(name: str, wave: np.ndarray):
        path = tmp_path / name
        sf.write(path, wave, SR)
        return path

    return _write
