"""Match per-file cluster embeddings against registry centroids.

Thresholds are backend-specific and corpus-specific; the defaults here are a
conservative starting point for ECAPA cosine similarity. `speakerdex process`
exposes both as flags, and threshold calibration from confirmed assignments is
on the roadmap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import MATCHED, NEW, REVIEW, MatchDecision


@dataclass
class MatchConfig:
    match_threshold: float = 0.55
    review_threshold: float = 0.40


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # inputs are L2-normalized


def match_clusters(
    cluster_embeddings: dict[str, tuple[np.ndarray, float]],
    centroids: dict[int, np.ndarray],
    names: dict[int, str],
    config: MatchConfig | None = None,
) -> list[MatchDecision]:
    """Greedy one-to-one matching of clusters to identities.

    cluster_embeddings: cluster label -> (embedding, seconds of speech)
    centroids: identity id -> centroid embedding
    names: identity id -> display name

    Within one file, two clusters cannot both claim the same identity: pairs
    are taken in order of descending similarity and an identity is consumed
    once matched (the diarizer already decided these are different voices).
    """
    config = config or MatchConfig()

    # Score every (cluster, identity) pair
    pairs: list[tuple[float, str, int]] = []
    for cluster, (emb, _dur) in cluster_embeddings.items():
        for identity_id, centroid in centroids.items():
            pairs.append((cosine(emb, centroid), cluster, identity_id))
    pairs.sort(key=lambda p: p[0], reverse=True)

    best_sim: dict[str, float] = {c: -1.0 for c in cluster_embeddings}
    best_id: dict[str, int | None] = {c: None for c in cluster_embeddings}
    taken: set[int] = set()
    assigned: set[str] = set()

    for sim, cluster, identity_id in pairs:
        if cluster in assigned or identity_id in taken:
            # still track the best raw similarity for reporting
            if sim > best_sim[cluster] and cluster not in assigned:
                pass
            continue
        if sim < config.review_threshold:
            break  # sorted desc: nothing below this point can match
        best_sim[cluster] = sim
        best_id[cluster] = identity_id
        assigned.add(cluster)
        taken.add(identity_id)

    decisions = []
    for cluster, (_emb, dur) in cluster_embeddings.items():
        identity_id = best_id[cluster]
        sim = best_sim[cluster]
        if identity_id is None:
            # report the closest identity anyway, for context in `review`
            closest = max(
                (p for p in pairs if p[1] == cluster), default=None, key=lambda p: p[0]
            )
            decisions.append(
                MatchDecision(
                    cluster=cluster,
                    identity_id=None,
                    identity_name=names.get(closest[2]) if closest else None,
                    similarity=closest[0] if closest else -1.0,
                    status=NEW,
                    total_speech=dur,
                )
            )
        else:
            status = MATCHED if sim >= config.match_threshold else REVIEW
            decisions.append(
                MatchDecision(
                    cluster=cluster,
                    identity_id=identity_id,
                    identity_name=names[identity_id],
                    similarity=sim,
                    status=status,
                    total_speech=dur,
                )
            )
    return decisions
