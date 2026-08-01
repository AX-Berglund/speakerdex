"""Threshold calibration from labelled voice samples.

Given a directory of known voices:

    voices/
      alex/   ep1_solo.wav  ep2_solo.wav  interview.wav
      sam/    intro.wav     outro.wav
      jordan/ clip1.wav     clip2.wav

this measures how the embedding backend actually behaves on *your* audio:
the distribution of same-speaker vs different-speaker cosine similarities,
recommended match/review thresholds, and leave-one-out identification
accuracy at those thresholds.

Subdirectories starting with "_" or "." are not speakers and are skipped, so
scratch folders can live alongside the voices (scripts/setup_voices.py caches
its MP3 downloads in voices/_downloads/).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import load_audio, select_chunks
from .embeddings import EmbeddingBackend, combine
from .types import Segment

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"}


def is_speaker_dir(path: Path) -> bool:
    """Speaker directories are visible directories; "_"/"." prefixes are scratch."""
    return path.is_dir() and not path.name.startswith(("_", "."))


def embed_file(
    path: str | Path, backend: EmbeddingBackend, max_total_sec: float = 60.0
) -> np.ndarray | None:
    """One embedding for a whole file (assumed to be a single speaker)."""
    wave, sr = load_audio(path)
    segments = [Segment(0.0, len(wave) / sr, "CALIB")]
    chunks = select_chunks(wave, sr, segments, max_total_sec=max_total_sec)
    if not chunks:
        return None
    embeddings = [backend.embed(chunk, sr) for chunk in chunks]
    durations = [len(chunk) / sr for chunk in chunks]
    return combine(embeddings, durations)


@dataclass
class CalibrationReport:
    n_speakers: int
    n_files: int
    same: list[float]  # same-speaker pairwise similarities
    diff: list[float]  # different-speaker pairwise similarities
    match_threshold: float
    review_threshold: float
    top1_accuracy: float
    top1_total: int
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        same, diff = np.array(self.same), np.array(self.diff)
        lines = [
            f"Speakers: {self.n_speakers}   Files: {self.n_files}",
            f"Same-speaker pairs: {len(same)}   Different-speaker pairs: {len(diff)}",
            "",
            f"same-speaker similarity:  min={same.min():.3f}  "
            f"p5={np.percentile(same, 5):.3f}  median={np.median(same):.3f}  "
            f"max={same.max():.3f}",
            f"diff-speaker similarity:  min={diff.min():.3f}  "
            f"median={np.median(diff):.3f}  p99={np.percentile(diff, 99):.3f}  "
            f"max={diff.max():.3f}",
            "",
            _histogram(same, diff),
            "",
            f"Recommended:  --match-threshold {self.match_threshold:.2f}  "
            f"--review-threshold {self.review_threshold:.2f}",
            f"Leave-one-out identification: {int(self.top1_accuracy * self.top1_total)}"
            f"/{self.top1_total} correct ({self.top1_accuracy:.0%})",
        ]
        for w in self.warnings:
            lines.append(f"WARNING: {w}")
        return "\n".join(lines)


def _histogram(same: np.ndarray, diff: np.ndarray, width: int = 40) -> str:
    """Two aligned ASCII histograms over cosine similarity in [-0.2, 1.0]."""
    bins = np.linspace(-0.2, 1.0, 25)
    same_counts, _ = np.histogram(same, bins=bins)
    diff_counts, _ = np.histogram(diff, bins=bins)
    peak = max(1, same_counts.max(), diff_counts.max())
    rows = ["         diff (x)                same (o)"]
    for i in range(len(bins) - 1):
        if same_counts[i] == 0 and diff_counts[i] == 0:
            continue
        bar_d = "x" * int(np.ceil(diff_counts[i] / peak * width))
        bar_s = "o" * int(np.ceil(same_counts[i] / peak * width))
        rows.append(f"{bins[i]:+.2f} | {bar_d}{bar_s}")
    return "\n".join(rows)


def _recommend(same: np.ndarray, diff: np.ndarray) -> tuple[float, float, list[str]]:
    warnings: list[str] = []
    same_lo = float(np.percentile(same, 5))
    diff_hi = float(np.percentile(diff, 99))
    separation = same_lo - diff_hi

    if separation > 0:
        match = diff_hi + 0.60 * separation
        review = diff_hi + 0.25 * separation
    else:
        warnings.append(
            "same- and different-speaker similarities overlap "
            f"(p5 same = {same_lo:.3f} <= p99 diff = {diff_hi:.3f}); expect review-band "
            "traffic and misses. More/cleaner enrollment audio usually helps."
        )
        match = float(np.median(same))
        review = max(float(np.percentile(diff, 90)), match - 0.15)

    match = round(min(max(match, review + 0.01), 0.99), 2)
    review = round(min(review, match - 0.01), 2)
    return match, review, warnings


def calibrate(
    voices_dir: str | Path,
    backend: EmbeddingBackend,
    max_total_sec: float = 60.0,
) -> CalibrationReport:
    voices_dir = Path(voices_dir)
    speakers: dict[str, list[np.ndarray]] = {}
    n_files = 0
    skipped: list[str] = []

    for speaker_dir in sorted(p for p in voices_dir.iterdir() if is_speaker_dir(p)):
        embeddings = []
        for audio_path in sorted(speaker_dir.iterdir()):
            if audio_path.suffix.lower() not in AUDIO_EXTS:
                continue
            emb = embed_file(audio_path, backend, max_total_sec=max_total_sec)
            if emb is None:
                skipped.append(str(audio_path))
                continue
            embeddings.append(emb)
            n_files += 1
        if len(embeddings) >= 2:
            speakers[speaker_dir.name] = embeddings
        elif embeddings:
            speakers[speaker_dir.name] = embeddings  # usable for diff-pairs only

    if len(speakers) < 2:
        raise ValueError(
            f"need at least 2 speaker directories with audio under {voices_dir} "
            "(layout: voices/<speaker_name>/*.wav; '_' and '.' prefixed dirs are skipped)"
        )

    names = sorted(speakers)
    same = [
        float(np.dot(a, b))
        for name in names
        for i, a in enumerate(speakers[name])
        for b in speakers[name][i + 1 :]
    ]
    diff = [
        float(np.dot(a, b))
        for i, n1 in enumerate(names)
        for n2 in names[i + 1 :]
        for a in speakers[n1]
        for b in speakers[n2]
    ]
    if not same:
        raise ValueError("no speaker has 2+ files; same-speaker statistics are impossible")

    match, review, warnings = _recommend(np.array(same), np.array(diff))
    if skipped:
        warnings.append(f"skipped {len(skipped)} file(s) with no usable speech: {skipped[:3]}")

    # Leave-one-out top-1 identification at the recommended thresholds
    correct = total = 0
    for name in names:
        embs = speakers[name]
        if len(embs) < 2:
            continue
        for i, held_out in enumerate(embs):
            centroids = {
                other: combine(
                    [e for j, e in enumerate(speakers[other]) if other != name or j != i]
                )
                for other in names
                if (len(speakers[other]) - (1 if other == name else 0)) >= 1
            }
            sims = {other: float(np.dot(held_out, c)) for other, c in centroids.items()}
            predicted = max(sims, key=sims.get)  # type: ignore[arg-type]
            total += 1
            if predicted == name and sims[predicted] >= review:
                correct += 1

    return CalibrationReport(
        n_speakers=len(speakers),
        n_files=n_files,
        same=same,
        diff=diff,
        match_threshold=match,
        review_threshold=review,
        top1_accuracy=(correct / total) if total else 0.0,
        top1_total=total,
        warnings=warnings,
    )
