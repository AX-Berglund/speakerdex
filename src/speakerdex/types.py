"""Core datatypes shared across speakerdex."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    """A diarized speech segment: cluster label + time span within one file."""

    start: float
    end: float
    speaker: str  # per-file cluster label from the diarizer, e.g. "SPEAKER_00"
    text: str = ""  # transcript, when the source format carries one (WhisperX)

    @property
    def duration(self) -> float:
        return self.end - self.start


def cluster_seconds(segments: list[Segment]) -> list[tuple[str, float]]:
    """(cluster label, total speech seconds) for each cluster, longest first."""
    totals: dict[str, float] = {}
    for seg in segments:
        totals[seg.speaker] = totals.get(seg.speaker, 0.0) + seg.duration
    return sorted(totals.items(), key=lambda kv: kv[1], reverse=True)


# Assignment status values
MATCHED = "matched"  # similarity >= match_threshold: confidently the enrolled identity
REVIEW = "review"  # between thresholds: probable match, needs human confirmation
NEW = "new"  # below review_threshold: treated as a previously unseen speaker


NO_SIMILARITY = -1.0
"""Sentinel: there was nothing to compare this cluster against (empty registry)."""


def similarity_json(similarity: float) -> float | None:
    """Similarity for JSON output; null when there was nothing to compare against."""
    return None if similarity < 0 else round(similarity, 4)


@dataclass
class MatchDecision:
    """Outcome of matching one per-file speaker cluster against the registry."""

    cluster: str
    identity_id: int | None
    identity_name: str | None
    similarity: float  # NO_SIMILARITY when the registry held no candidates
    status: str  # MATCHED | REVIEW | NEW
    total_speech: float = 0.0  # seconds of speech the cluster embedding was computed from

    def __str__(self) -> str:
        name = self.identity_name or "?"
        sim = "n/a" if self.similarity < 0 else f"{self.similarity:.3f}"
        # How much speech backed the decision is central to trusting it. Omitted
        # when unknown, e.g. replayed from a stored assignment.
        speech = f", {self.total_speech:.0f}s" if self.total_speech > 0 else ""
        return f"{self.cluster} -> {name} ({self.status}, sim={sim}{speech})"
