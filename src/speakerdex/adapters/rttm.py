"""RTTM (Rich Transcription Time Marked) adapter — the diarization lingua franca.

Line format (SPEAKER records):
SPEAKER <file-id> <chan> <onset> <duration> <NA> <NA> <speaker> <NA> <NA>
"""

from __future__ import annotations

from pathlib import Path

from ..types import Segment


def load_rttm(path: str | Path) -> list[Segment]:
    segments: list[Segment] = []
    for lineno, line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = line.strip()
        if not line or not line.startswith("SPEAKER"):
            continue
        parts = line.split()
        if len(parts) < 8:
            raise ValueError(f"{path}:{lineno}: malformed RTTM line: {line!r}")
        onset, duration, speaker = float(parts[3]), float(parts[4]), parts[7]
        segments.append(Segment(start=onset, end=onset + duration, speaker=speaker))
    return segments


def write_rttm(
    segments: list[Segment],
    path: str | Path,
    file_id: str = "audio",
    mapping: dict[str, str] | None = None,
) -> None:
    """Write segments back out, optionally relabelling clusters to identity names."""
    mapping = mapping or {}
    lines = []
    for seg in segments:
        speaker = mapping.get(seg.speaker, seg.speaker).replace(" ", "_")
        lines.append(
            f"SPEAKER {file_id} 1 {seg.start:.3f} {seg.duration:.3f} "
            f"<NA> <NA> {speaker} <NA> <NA>"
        )
    Path(path).write_text("\n".join(lines) + "\n")
