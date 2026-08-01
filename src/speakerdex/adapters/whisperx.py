"""WhisperX-style JSON adapter.

Expected shape: {"segments": [{"start": ..., "end": ..., "speaker": "SPEAKER_00", ...}, ...]}
Segments without a "speaker" key are skipped (WhisperX leaves them unlabelled
when diarization is unsure).
"""

from __future__ import annotations

import json
from pathlib import Path

from ..types import Segment


def load_whisperx(path: str | Path) -> list[Segment]:
    data = json.loads(Path(path).read_text())
    segments: list[Segment] = []
    for seg in data.get("segments", []):
        speaker = seg.get("speaker")
        if speaker is None:
            continue
        segments.append(Segment(start=float(seg["start"]), end=float(seg["end"]), speaker=speaker))
    return segments


def write_whisperx(
    in_path: str | Path, out_path: str | Path, mapping: dict[str, str]
) -> None:
    """Copy a WhisperX JSON, relabelling speaker fields to identity names.

    The original cluster label is preserved as "speaker_cluster" so the
    relabelling is reversible.
    """
    data = json.loads(Path(in_path).read_text())
    for seg in data.get("segments", []):
        cluster = seg.get("speaker")
        if cluster in mapping:
            seg["speaker_cluster"] = cluster
            seg["speaker"] = mapping[cluster]
        # word-level speakers, if present
        for word in seg.get("words", []):
            wcluster = word.get("speaker")
            if wcluster in mapping:
                word["speaker"] = mapping[wcluster]
    Path(out_path).write_text(json.dumps(data, ensure_ascii=False, indent=2))
