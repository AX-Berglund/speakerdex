import numpy as np

from speakerdex.matcher import MatchConfig, match_clusters
from speakerdex.types import MATCHED, NEW, REVIEW


def one_hot(i: int, dim: int = 8) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


def blend(i: int, j: int, cos: float, dim: int = 8) -> np.ndarray:
    """Unit vector whose cosine similarity with one_hot(i) is exactly `cos`."""
    return cos * one_hot(i, dim) + np.sqrt(1 - cos**2) * one_hot(j, dim)


NAMES = {1: "Alice", 2: "Bob"}


def test_confident_match_and_new():
    centroids = {1: one_hot(0), 2: one_hot(1)}
    clusters = {
        "SPEAKER_00": (one_hot(0), 30.0),  # exactly Alice
        "SPEAKER_01": (one_hot(5), 12.0),  # nobody we know
    }
    decisions = {d.cluster: d for d in match_clusters(clusters, centroids, NAMES)}
    assert decisions["SPEAKER_00"].status == MATCHED
    assert decisions["SPEAKER_00"].identity_name == "Alice"
    assert decisions["SPEAKER_01"].status == NEW
    assert decisions["SPEAKER_01"].identity_id is None


def test_review_band():
    centroids = {1: one_hot(0)}
    # similarity ~0.45: between review (0.40) and match (0.55) thresholds
    clusters = {"SPEAKER_00": (blend(0, 3, 0.45), 10.0)}
    sim = float(np.dot(clusters["SPEAKER_00"][0], centroids[1]))
    assert 0.40 < sim < 0.55
    d = match_clusters(clusters, centroids, NAMES)[0]
    assert d.status == REVIEW
    assert d.identity_name == "Alice"


def test_one_to_one_within_file():
    """Two clusters can't both claim the same identity; best pair wins."""
    centroids = {1: one_hot(0)}
    clusters = {
        "SPEAKER_00": (blend(0, 1, 0.95), 30.0),  # very close to Alice
        "SPEAKER_01": (blend(0, 1, 0.80), 30.0),  # close-ish to Alice too
    }
    decisions = {d.cluster: d for d in match_clusters(clusters, centroids, NAMES)}
    assert decisions["SPEAKER_00"].identity_name == "Alice"
    assert decisions["SPEAKER_00"].status == MATCHED
    # the weaker claimant loses Alice and is treated as new
    assert decisions["SPEAKER_01"].identity_id is None
    assert decisions["SPEAKER_01"].status == NEW


def test_empty_registry_everything_is_new():
    clusters = {"SPEAKER_00": (one_hot(0), 5.0)}
    d = match_clusters(clusters, {}, {})[0]
    assert d.status == NEW
    assert d.identity_id is None


def test_custom_thresholds():
    centroids = {1: one_hot(0)}
    clusters = {"SPEAKER_00": (blend(0, 3, 0.45), 10.0)}
    config = MatchConfig(match_threshold=0.3, review_threshold=0.2)
    d = match_clusters(clusters, centroids, NAMES, config)[0]
    assert d.status == MATCHED
