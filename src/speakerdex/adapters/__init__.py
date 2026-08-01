"""Input adapters: parse diarization output formats into [Segment]."""

from __future__ import annotations

from pathlib import Path

from ..types import Segment
from .rttm import load_rttm, write_rttm
from .whisperx import load_whisperx, write_whisperx

__all__ = ["load_rttm", "write_rttm", "load_whisperx", "write_whisperx", "load_segments"]


def load_segments(path: str | Path, fmt: str | None = None) -> list[Segment]:
    """Load segments, inferring the format from the extension when not given."""
    path = Path(path)
    if fmt is None:
        fmt = {".rttm": "rttm", ".json": "whisperx"}.get(path.suffix.lower())
        if fmt is None:
            raise ValueError(f"cannot infer diarization format from {path.name!r}; pass --format")
    if fmt == "rttm":
        return load_rttm(path)
    if fmt == "whisperx":
        return load_whisperx(path)
    raise ValueError(f"unknown diarization format {fmt!r} (expected 'rttm' or 'whisperx')")
