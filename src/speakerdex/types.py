"""Core datatypes shared across speakerdex."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    """A diarized speech segment: cluster label + time span within one file."""

    start: float
    end: float
    speaker: str  # per-file cluster label from the diarizer, e.g. "SPEAKER_00"

    @property
    def duration(self) -> float:
        return self.end - self.start


# Assignment status values
MATCHED = "matched"  # similarity >= match_threshold: confidently the enrolled identity
REVIEW = "review"  # between thresholds: probable match, needs human confirmation
NEW = "new"  # below review_threshold: treated as a previously unseen speaker


@dataclass
class MatchDecision:
    """Outcome of matching one per-file speaker cluster against the registry."""

    cluster: str
    identity_id: int | None
    identity_name: str | None
    similarity: float
    status: str  # MATCHED | REVIEW | NEW
    total_speech: float = 0.0  # seconds of speech the cluster embedding was computed from

    def __str__(self) -> str:
        name = self.identity_name or "?"
        return f"{self.cluster} -> {name} ({self.status}, sim={self.similarity:.3f})"
