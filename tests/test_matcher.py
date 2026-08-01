import numpy as np
import pytest

from speakerdex.matcher import MatchConfig, match_clusters
from speakerdex.types import MATCHED, NEW, NO_SIMILARITY, REVIEW, similarity_json


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


def test_new_decision_reports_closest_identity_for_context():
    """A NEW cluster still names its nearest identity, so `review` has context."""
    centroids = {1: one_hot(0), 2: one_hot(1)}
    # closer to Bob than Alice, but below the review threshold for both
    clusters = {"SPEAKER_00": (blend(1, 5, 0.30), 8.0)}
    d = match_clusters(clusters, centroids, NAMES)[0]
    assert d.status == NEW
    assert d.identity_id is None  # not an assignment, just context
    assert d.identity_name == "Bob"
    assert d.similarity == pytest.approx(0.30)


def test_loser_of_one_to_one_reports_the_identity_it_lost():
    """The cluster beaten to an identity still reports it as its closest."""
    centroids = {1: one_hot(0)}
    clusters = {
        "SPEAKER_00": (blend(0, 1, 0.95), 30.0),
        "SPEAKER_01": (blend(0, 1, 0.80), 30.0),
    }
    decisions = {d.cluster: d for d in match_clusters(clusters, centroids, NAMES)}
    loser = decisions["SPEAKER_01"]
    assert loser.status == NEW
    assert loser.identity_id is None
    assert loser.identity_name == "Alice"
    assert loser.similarity == pytest.approx(0.80)  # its raw score, not the winner's


def test_empty_registry_everything_is_new():
    clusters = {"SPEAKER_00": (one_hot(0), 5.0)}
    d = match_clusters(clusters, {}, {})[0]
    assert d.status == NEW
    assert d.identity_id is None
    assert d.identity_name is None
    assert d.similarity == NO_SIMILARITY  # nothing to compare against


def test_no_similarity_renders_as_na_not_minus_one():
    d = match_clusters({"SPEAKER_00": (one_hot(0), 5.0)}, {}, {})[0]
    assert "sim=n/a" in str(d)
    assert "-1.000" not in str(d)
    assert similarity_json(d.similarity) is None
    # a real score is still a number
    real = match_clusters({"SPEAKER_00": (one_hot(0), 5.0)}, {1: one_hot(0)}, NAMES)[0]
    assert "sim=1.000" in str(real)
    assert similarity_json(real.similarity) == 1.0


def test_custom_thresholds():
    centroids = {1: one_hot(0)}
    clusters = {"SPEAKER_00": (blend(0, 3, 0.45), 10.0)}
    config = MatchConfig(match_threshold=0.3, review_threshold=0.2)
    d = match_clusters(clusters, centroids, NAMES, config)[0]
    assert d.status == MATCHED
