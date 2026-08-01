"""End-to-end processing: audio + diarization -> stable identities.

This is the library-level entrypoint the CLI wraps.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import MAX_TOTAL_SEC, MIN_SEGMENT_SEC, load_audio, select_chunks
from .embeddings import EmbeddingBackend, combine
from .matcher import MatchConfig, match_clusters
from .registry import Registry
from .types import NEW, MatchDecision, Segment


@dataclass
class ProcessConfig:
    match: MatchConfig = field(default_factory=MatchConfig)
    min_segment_sec: float = MIN_SEGMENT_SEC
    max_total_sec: float = MAX_TOTAL_SEC
    enroll_unknowns: bool = False  # auto-create "Unknown-N" identities for NEW clusters
    reinforce: bool = False  # add high-confidence cluster embeddings as new voiceprints


def embed_clusters(
    wave: np.ndarray,
    sr: int,
    segments: list[Segment],
    backend: EmbeddingBackend,
    config: ProcessConfig | None = None,
) -> dict[str, tuple[np.ndarray, float]]:
    """Compute one embedding per cluster label: cluster -> (embedding, speech seconds)."""
    config = config or ProcessConfig()
    by_cluster: dict[str, list[Segment]] = defaultdict(list)
    for seg in segments:
        by_cluster[seg.speaker].append(seg)

    result: dict[str, tuple[np.ndarray, float]] = {}
    for cluster, segs in by_cluster.items():
        chunks = select_chunks(
            wave,
            sr,
            segs,
            min_segment_sec=config.min_segment_sec,
            max_total_sec=config.max_total_sec,
        )
        if not chunks:
            continue  # cluster has no usable speech (all blips)
        embeddings = [backend.embed(chunk, sr) for chunk in chunks]
        durations = [len(chunk) / sr for chunk in chunks]
        result[cluster] = (combine(embeddings, durations), sum(durations))
    return result


def _next_unknown_name(registry: Registry) -> str:
    existing = {i.name for i in registry.identities()}
    n = 1
    while f"Unknown-{n}" in existing:
        n += 1
    return f"Unknown-{n}"


def process_file(
    audio_path: str | Path,
    segments: list[Segment],
    registry: Registry,
    backend: EmbeddingBackend,
    config: ProcessConfig | None = None,
) -> list[MatchDecision]:
    """Match every speaker cluster in one file against the registry.

    Records assignments in the registry and (optionally) enrolls unknowns /
    reinforces matched identities with fresh voiceprints.
    """
    config = config or ProcessConfig()
    registry.check_backend(backend.name)
    source = str(audio_path)

    wave, sr = load_audio(audio_path)
    cluster_embeddings = embed_clusters(wave, sr, segments, backend, config)

    identities = registry.identities()
    names = {i.id: i.name for i in identities}
    decisions = match_clusters(cluster_embeddings, registry.centroids(), names, config.match)

    for decision in decisions:
        emb, dur = cluster_embeddings[decision.cluster]
        if decision.status == NEW and config.enroll_unknowns:
            name = _next_unknown_name(registry)
            identity_id = registry.enroll(name, notes=f"auto-enrolled from {source}")
            registry.add_voiceprint(identity_id, emb, duration=dur, source=source)
            decision.identity_id = identity_id
            decision.identity_name = name
        elif decision.status == "matched" and config.reinforce and decision.identity_id:
            registry.add_voiceprint(decision.identity_id, emb, duration=dur, source=source)
        registry.record_assignment(
            source, decision.cluster, decision.identity_id, decision.similarity, decision.status
        )
    return decisions


def enroll_from_audio(
    name: str,
    audio_path: str | Path,
    registry: Registry,
    backend: EmbeddingBackend,
    segments: list[Segment] | None = None,
    cluster: str | None = None,
    config: ProcessConfig | None = None,
) -> int:
    """Enroll an identity from audio.

    With no segments: the whole file is assumed to be one speaker talking.
    With segments + cluster: only that cluster's speech is used (enroll
    someone directly out of a diarized episode).
    """
    config = config or ProcessConfig()
    registry.check_backend(backend.name)
    wave, sr = load_audio(audio_path)

    if segments is not None:
        if cluster is None:
            raise ValueError("when segments are given, a cluster label is required")
        segs = [s for s in segments if s.speaker == cluster]
        if not segs:
            raise ValueError(f"no segments with cluster label {cluster!r}")
    else:
        segs = [Segment(start=0.0, end=len(wave) / sr, speaker="ENROLL")]

    chunks = select_chunks(
        wave, sr, segs, min_segment_sec=config.min_segment_sec, max_total_sec=config.max_total_sec
    )
    if not chunks:
        raise ValueError("no usable speech found to enroll from")
    embeddings = [backend.embed(chunk, sr) for chunk in chunks]
    durations = [len(chunk) / sr for chunk in chunks]
    emb = combine(embeddings, durations)

    ident = registry.identity_by_name(name)
    identity_id = ident.id if ident else registry.enroll(name)
    registry.add_voiceprint(identity_id, emb, duration=sum(durations), source=str(audio_path))
    return identity_id
