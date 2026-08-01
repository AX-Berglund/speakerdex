"""End-to-end: enroll a voice from one file, recognize it in another."""

import pytest

from speakerdex.pipeline import ProcessConfig, enroll_from_audio, process_file
from speakerdex.registry import Registry
from speakerdex.types import MATCHED, NEW, Segment

from .conftest import build_track

ALICE = 220.0  # fundamental frequencies of our two synthetic "voices"
BOB = 520.0


@pytest.fixture()
def registry(tmp_path):
    with Registry(tmp_path / "reg.db") as reg:
        yield reg


def test_enroll_then_recognize_across_files(registry, backend, write_wav):
    # File 1: Alice alone (a clean enrollment clip)
    enroll_wav = write_wav("alice_solo.wav", build_track([(ALICE, 6.0)]))
    enroll_from_audio("Alice", enroll_wav, registry, backend)
    assert [i.name for i in registry.identities()] == ["Alice"]

    # File 2: a "conversation": Alice speaks, silence, Bob speaks, Alice again
    episode = build_track([(ALICE, 4.0), (None, 1.0), (BOB, 4.0), (ALICE, 3.0)], seed=42)
    ep_wav = write_wav("ep1.wav", episode)
    segments = [
        Segment(0.0, 4.0, "SPEAKER_00"),
        Segment(5.0, 9.0, "SPEAKER_01"),
        Segment(9.0, 12.0, "SPEAKER_00"),
    ]
    decisions = {d.cluster: d for d in process_file(ep_wav, segments, registry, backend)}

    assert decisions["SPEAKER_00"].status == MATCHED
    assert decisions["SPEAKER_00"].identity_name == "Alice"
    assert decisions["SPEAKER_01"].status == NEW

    # assignments were recorded
    rows = registry.conn.execute(
        "SELECT cluster, status FROM assignments ORDER BY cluster"
    ).fetchall()
    assert rows == [("SPEAKER_00", MATCHED), ("SPEAKER_01", NEW)]


def test_enroll_unknowns_then_recognize_them_later(registry, backend, write_wav):
    config = ProcessConfig(enroll_unknowns=True)

    # Episode 1: two strangers
    ep1 = build_track([(ALICE, 5.0), (BOB, 5.0)])
    ep1_wav = write_wav("ep1.wav", ep1)
    segs1 = [Segment(0.0, 5.0, "SPEAKER_00"), Segment(5.0, 10.0, "SPEAKER_01")]
    decisions1 = process_file(ep1_wav, segs1, registry, backend, config)
    assert all(d.identity_name and d.identity_name.startswith("Unknown-") for d in decisions1)
    assert len(registry.identities()) == 2

    # Episode 2: the same two voices, different cluster labels and order
    ep2 = build_track([(BOB, 5.0), (ALICE, 5.0)], seed=7)
    ep2_wav = write_wav("ep2.wav", ep2)
    segs2 = [Segment(0.0, 5.0, "SPEAKER_00"), Segment(5.0, 10.0, "SPEAKER_01")]
    decisions2 = {d.cluster: d for d in process_file(ep2_wav, segs2, registry, backend, config)}

    # Bob spoke first in ep2: SPEAKER_00 must resolve to the identity ep1 gave Bob
    ep1_by_cluster = {d.cluster: d.identity_name for d in decisions1}
    assert decisions2["SPEAKER_00"].identity_name == ep1_by_cluster["SPEAKER_01"]
    assert decisions2["SPEAKER_01"].identity_name == ep1_by_cluster["SPEAKER_00"]
    assert all(d.status == MATCHED for d in decisions2.values())
    assert len(registry.identities()) == 2  # nothing new was enrolled


def test_enroll_from_diarized_cluster(registry, backend, write_wav):
    episode = build_track([(ALICE, 5.0), (BOB, 5.0)])
    wav = write_wav("ep.wav", episode)
    segments = [Segment(0.0, 5.0, "SPEAKER_00"), Segment(5.0, 10.0, "SPEAKER_01")]
    enroll_from_audio("Bob", wav, registry, backend, segments=segments, cluster="SPEAKER_01")

    decisions = {d.cluster: d for d in process_file(wav, segments, registry, backend)}
    assert decisions["SPEAKER_01"].identity_name == "Bob"
    assert decisions["SPEAKER_01"].status == MATCHED
    assert decisions["SPEAKER_00"].status == NEW


def test_bad_cluster_error_lists_the_real_clusters(registry, backend, write_wav):
    """The error must be enough to fix the command; no second lookup needed."""
    wav = write_wav("ep.wav", build_track([(ALICE, 5.0), (BOB, 3.0)]))
    segments = [Segment(0.0, 5.0, "SPEAKER_00"), Segment(5.0, 8.0, "SPEAKER_01")]

    with pytest.raises(ValueError) as excinfo:
        enroll_from_audio("Bob", wav, registry, backend, segments=segments, cluster="SPEAKER_99")

    message = str(excinfo.value)
    assert "SPEAKER_99" in message
    assert "2 cluster(s)" in message
    assert "SPEAKER_00  5.0s" in message  # longest first, with seconds
    assert "SPEAKER_01  3.0s" in message


def test_enroll_reports_seconds_captured(registry, backend, write_wav):
    wav = write_wav("alice.wav", build_track([(ALICE, 6.0)]))
    result = enroll_from_audio("Alice", wav, registry, backend)
    assert result.seconds == pytest.approx(6.0, abs=0.1)
    assert result.chunks == 1
    assert result.is_thin  # 6s is under the 10s floor
    assert registry.identity_by_name("Alice").id == result.identity_id

    longer = write_wav("alice2.wav", build_track([(ALICE, 20.0)]))
    assert not enroll_from_audio("Alice", longer, registry, backend).is_thin


def test_reinforce_adds_voiceprints(registry, backend, write_wav):
    enroll_wav = write_wav("alice.wav", build_track([(ALICE, 5.0)]))
    enroll_from_audio("Alice", enroll_wav, registry, backend)
    ep_wav = write_wav("ep.wav", build_track([(ALICE, 6.0)], seed=3))
    segments = [Segment(0.0, 6.0, "SPEAKER_00")]

    process_file(ep_wav, segments, registry, backend, ProcessConfig(reinforce=True))
    assert registry.identities()[0].voiceprint_count == 2


def test_backend_mismatch_refused(registry, backend, write_wav):
    registry.set_meta("backend", "ecapa")
    wav = write_wav("a.wav", build_track([(ALICE, 2.0)]))
    with pytest.raises(ValueError, match="refusing to mix"):
        process_file(wav, [Segment(0.0, 2.0, "S0")], registry, backend)
