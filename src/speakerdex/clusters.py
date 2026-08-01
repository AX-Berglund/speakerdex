"""Inspect the speaker clusters inside one diarization file.

`enroll --cluster SPEAKER_02` needs a cluster label, but a diarizer's labels are
opaque: nothing about "SPEAKER_02" says whether it is the host, the guest, or
the pre-roll ad. This closes that gap — how long each cluster speaks, and, when
the diarization carries a transcript, what it actually said.

With audio and a non-empty registry it also runs a read-only dry run of
`process`: each cluster is embedded and scored against the registry, so you can
see who speakerdex *would* say each cluster is before enrolling anything. No
identities, voiceprints or assignments are written.

Note that "the longest cluster is the host" is a bad heuristic on interview
formats — a guest routinely outspeaks the host two to one. Read the previews.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .adapters import load_segments
from .embeddings import EmbeddingBackend
from .matcher import MatchConfig, match_clusters
from .registry import Registry
from .types import NEW, MatchDecision, Segment, cluster_seconds

PREVIEW_WORDS = 12
MIN_PREVIEW_WORDS = 5  # skip "What do you think?" — interjections identify nobody


def _timestamp(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def _clip(text: str) -> str:
    words = text.split()
    return " ".join(words[:PREVIEW_WORDS]) + ("…" if len(words) > PREVIEW_WORDS else "")


@dataclass
class Preview:
    """A short quotation from one segment, for recognizing who a cluster is."""

    start: float
    text: str

    def __str__(self) -> str:
        return f"[{_timestamp(self.start)}]  {_clip(self.text)}"


@dataclass
class ClusterInfo:
    label: str
    speech_seconds: float
    segment_count: int
    share: float  # fraction of all labelled speech in the file
    longest: Preview | None = None
    middle: Preview | None = None
    match: MatchDecision | None = None  # only with --audio and a non-empty registry

    def previews(self) -> list[Preview]:
        return [p for p in (self.longest, self.middle) if p is not None]


def _previews(segments: list[Segment], midpoint: float) -> tuple[Preview | None, Preview | None]:
    """The cluster's longest segment, plus one from around the middle of the file.

    Two samples beat one: the longest turn is usually the most substantive, but
    it is often an uninterrupted monologue that reads the same for everyone. A
    mid-file sample catches the conversational register too.
    """
    with_text = [s for s in segments if s.text.strip()]
    if not with_text:
        return None, None

    longest = max(with_text, key=lambda s: s.duration)
    rest = [s for s in with_text if s is not longest]
    if not rest:
        return Preview(longest.start, longest.text), None
    # Prefer a substantive turn near the midpoint; fall back to whatever is
    # nearest if this cluster only ever says short things.
    substantive = [s for s in rest if len(s.text.split()) >= MIN_PREVIEW_WORDS]
    middle = min(substantive or rest, key=lambda s: abs(s.start - midpoint))
    return Preview(longest.start, longest.text), Preview(middle.start, middle.text)


def _match_text(match: MatchDecision) -> str:
    """Render a dry-run verdict. A NEW cluster has no likely identity — saying
    "likely: X" of the identity it merely came closest to would be a lie."""
    sim = "n/a" if match.similarity < 0 else f"sim={match.similarity:.3f}"
    name = match.identity_name
    if match.status == NEW:
        return f"no match ({f'closest: {name}, ' if name else ''}{sim})"
    return f"likely: {name} ({sim}, {match.status})"


@dataclass
class ClusterReport:
    diarization: Path
    audio: Path | None
    clusters: list[ClusterInfo]
    total_seconds: float
    total_segments: int
    matched_against_registry: bool = False
    registry_empty: bool = False

    def summary(self) -> str:
        if not self.clusters:
            return f"No labelled speaker segments in {self.diarization}"

        lines = [
            f"{self.diarization.name}: {len(self.clusters)} clusters, "
            f"{self.total_segments} segments, {self.total_seconds:.1f}s labelled speech",
            "",
        ]
        width = max(len(c.label) for c in self.clusters)
        for info in self.clusters:
            row = (
                f"{info.label:<{width}}  {info.speech_seconds:7.1f}s  {info.share:5.1%}  "
                f"{info.segment_count:3d} seg"
            )
            if info.match is not None:
                row += "   " + _match_text(info.match)
            lines.append(row)
            for preview in info.previews():
                lines.append(f"    {preview}")
            if info.previews():
                lines.append("")

        if self.audio is not None and self.registry_empty:
            lines.append("Registry has no identities yet, so there is nothing to match against.")
        elif self.audio is None:
            lines.append("Pass --audio to also see the likely registry match for each cluster.")
        lines.append(
            f'Enroll with:  speakerdex enroll "<name>" <audio> '
            f"--diarization {self.diarization.name} --cluster <LABEL>"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "diarization": str(self.diarization),
            "audio": str(self.audio) if self.audio else None,
            "totals": {
                "clusters": len(self.clusters),
                "segments": self.total_segments,
                "speech_seconds": round(self.total_seconds, 2),
            },
            "clusters": [
                {
                    "cluster": c.label,
                    "speech_seconds": round(c.speech_seconds, 2),
                    "segments": c.segment_count,
                    "share": round(c.share, 4),
                    "previews": {
                        name: (
                            None
                            if preview is None
                            else {"start": round(preview.start, 2), "text": _clip(preview.text)}
                        )
                        for name, preview in (("longest", c.longest), ("middle", c.middle))
                    },
                    "match": (
                        None
                        if c.match is None
                        else {
                            "identity": c.match.identity_name,
                            "similarity": (
                                None if c.match.similarity < 0 else round(c.match.similarity, 4)
                            ),
                            "status": c.match.status,
                        }
                    ),
                }
                for c in self.clusters
            ],
        }


def inspect_clusters(
    diarization: str | Path,
    fmt: str | None = None,
    audio: str | Path | None = None,
    registry: Registry | None = None,
    backend: EmbeddingBackend | None = None,
    match: MatchConfig | None = None,
) -> ClusterReport:
    """Summarize every cluster in a diarization file, longest-speaking first.

    Pass audio + registry + backend to additionally dry-run the match. This is
    strictly read-only: it never enrolls, reinforces or records assignments.
    """
    diarization = Path(diarization)
    segments = load_segments(diarization, fmt)

    totals = cluster_seconds(segments)
    total_seconds = sum(secs for _, secs in totals)
    midpoint = max((s.end for s in segments), default=0.0) / 2

    decisions: dict[str, MatchDecision] = {}
    registry_empty = False
    if audio is not None and registry is not None:
        centroids = registry.centroids()
        registry_empty = not centroids
        if centroids and backend is not None:
            # Local import: pipeline imports this module's cluster helpers.
            from .audio import load_audio
            from .pipeline import embed_clusters

            wave, sr = load_audio(audio)
            embeddings = embed_clusters(wave, sr, segments, backend)
            names = {i.id: i.name for i in registry.identities()}
            decisions = {
                d.cluster: d for d in match_clusters(embeddings, centroids, names, match)
            }

    clusters = []
    for label, secs in totals:
        own = [s for s in segments if s.speaker == label]
        longest, middle = _previews(own, midpoint)
        clusters.append(
            ClusterInfo(
                label=label,
                speech_seconds=secs,
                segment_count=len(own),
                share=(secs / total_seconds) if total_seconds else 0.0,
                longest=longest,
                middle=middle,
                match=decisions.get(label),
            )
        )

    return ClusterReport(
        diarization=diarization,
        audio=Path(audio) if audio else None,
        clusters=clusters,
        total_seconds=total_seconds,
        total_segments=len(segments),
        matched_against_registry=bool(decisions),
        registry_empty=registry_empty,
    )
