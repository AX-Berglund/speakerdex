import numpy as np
import pytest

from speakerdex.registry import Registry


@pytest.fixture()
def registry(tmp_path):
    with Registry(tmp_path / "test.db") as reg:
        yield reg


def test_enroll_and_list(registry):
    registry.enroll("Alice")
    registry.enroll("Bob", notes="the host")
    names = [i.name for i in registry.identities()]
    assert names == ["Alice", "Bob"]
    assert registry.identity_by_name("Bob").notes == "the host"
    assert registry.identity_by_name("Carol") is None


def test_duplicate_name_rejected(registry):
    registry.enroll("Alice")
    import sqlite3

    with pytest.raises(sqlite3.IntegrityError):
        registry.enroll("Alice")


def test_voiceprints_and_centroids(registry):
    alice = registry.enroll("Alice")
    registry.add_voiceprint(alice, np.array([1.0, 0.0, 0.0]), duration=10)
    registry.add_voiceprint(alice, np.array([0.0, 1.0, 0.0]), duration=5)
    centroids = registry.centroids()
    assert set(centroids) == {alice}
    np.testing.assert_allclose(np.linalg.norm(centroids[alice]), 1.0, rtol=1e-6)
    # mean of e1 and e2, normalized -> equal weight on both axes
    np.testing.assert_allclose(centroids[alice][0], centroids[alice][1])
    assert registry.identities()[0].voiceprint_count == 2


def test_rename_and_merge(registry):
    a = registry.enroll("Unknown-1")
    b = registry.enroll("Alice")
    registry.add_voiceprint(a, np.array([1.0, 0.0]))
    registry.add_voiceprint(b, np.array([0.0, 1.0]))
    registry.rename("Unknown-1", "Sam")
    assert registry.identity_by_name("Sam") is not None

    registry.merge("Sam", "Alice")
    assert registry.identity_by_name("Sam") is None
    assert registry.identities()[0].voiceprint_count == 2
    with pytest.raises(KeyError):
        registry.rename("Sam", "Anyone")


def test_assignments_and_review_flow(registry):
    alice = registry.enroll("Alice")
    registry.record_assignment("ep1.wav", "SPEAKER_00", alice, 0.48, "review")
    registry.record_assignment("ep1.wav", "SPEAKER_01", None, 0.1, "new")
    reviews = registry.pending_reviews()
    assert reviews == [("ep1.wav", "SPEAKER_00", "Alice", 0.48)]

    registry.confirm("ep1.wav", "SPEAKER_00", "Alice")
    assert registry.pending_reviews() == []

    # upsert: reprocessing the same file replaces the assignment
    registry.record_assignment("ep1.wav", "SPEAKER_00", alice, 0.9, "matched")
    with pytest.raises(KeyError):
        registry.confirm("ep1.wav", "SPEAKER_99", "Alice")


def test_backend_lock(registry):
    registry.check_backend("ecapa")
    registry.check_backend("ecapa")  # same backend is fine
    with pytest.raises(ValueError, match="refusing to mix"):
        registry.check_backend("other-model")
