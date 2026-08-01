"""`speakerdex clusters`: know who is in a diarization before enrolling.

The dry-run tests use thresholds calibrated for the fake spectral backend
(same voice ~1.0, different ~0.67) rather than the ECAPA defaults — see
test_batch.py for why.
"""

import json

import pytest
import soundfile as sf
from typer.testing import CliRunner

from speakerdex.adapters.rttm import write_rttm
from speakerdex.cli import app
from speakerdex.clusters import PREVIEW_WORDS, inspect_clusters
from speakerdex.matcher import MatchConfig
from speakerdex.pipeline import enroll_from_audio
from speakerdex.registry import Registry
from speakerdex.types import MATCHED, NEW, Segment

from .conftest import SR, build_track

runner = CliRunner()

ALICE, BOB = 220.0, 520.0
FAKE = MatchConfig(match_threshold=0.90, review_threshold=0.80)
FAKE_FLAGS = ["--match-threshold", "0.90", "--review-threshold", "0.80"]

LONG_LINE = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"

# start, end, speaker, text
TURNS = [
    (0.0, 4.0, "SPEAKER_00", LONG_LINE),
    (5.0, 9.0, "SPEAKER_01", "guest talking at length about something entirely"),
    (10.0, 12.0, "SPEAKER_00", "the middle of the episode"),
    (20.0, 21.0, "SPEAKER_00", "short"),
]


def write_json(path, turns=TURNS):
    path.write_text(
        json.dumps(
            {"segments": [{"start": s, "end": e, "speaker": spk, "text": t}
                          for s, e, spk, t in turns]}
        )
    )
    return path


def write_rttm_file(path, turns=TURNS):
    write_rttm([Segment(s, e, spk) for s, e, spk, _ in turns], path, file_id="ep")
    return path


# -- counting --------------------------------------------------------------


def test_rttm_counts_but_has_no_previews(tmp_path):
    """RTTM carries no transcript, so there is nothing to preview."""
    report = inspect_clusters(write_rttm_file(tmp_path / "ep.rttm"))

    assert [c.label for c in report.clusters] == ["SPEAKER_00", "SPEAKER_01"]  # longest first
    assert report.total_segments == 4
    assert report.total_seconds == pytest.approx(11.0)

    top = report.clusters[0]
    assert top.speech_seconds == pytest.approx(7.0)
    assert top.segment_count == 3
    assert top.share == pytest.approx(7 / 11)
    assert top.longest is None and top.middle is None
    assert report.clusters[1].speech_seconds == pytest.approx(4.0)


def test_whisperx_previews_longest_and_middle(tmp_path):
    report = inspect_clusters(write_json(tmp_path / "ep.json"))
    top = report.clusters[0]

    # longest segment (0-4s) is previewed, clipped to PREVIEW_WORDS
    assert top.longest.start == pytest.approx(0.0)
    assert top.longest.text == LONG_LINE
    assert str(top.longest).endswith("…")
    assert "twelve" in str(top.longest) and "thirteen" not in str(top.longest)
    assert len(str(top.longest).split("]")[1].split()) == PREVIEW_WORDS

    # file midpoint is 10.5s, so the 10s turn is the mid-episode sample
    assert top.middle.start == pytest.approx(10.0)
    assert "middle of the episode" in str(top.middle)


def test_single_segment_cluster_has_no_middle(tmp_path):
    report = inspect_clusters(write_json(tmp_path / "ep.json"))
    guest = report.clusters[1]
    assert guest.longest is not None
    assert guest.middle is None  # only one segment: nothing else to sample


def test_no_labelled_segments(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"segments": [{"start": 0, "end": 1, "text": "(music)"}]}))
    report = inspect_clusters(path)
    assert report.clusters == []
    assert "No labelled speaker segments" in report.summary()


# -- dry run against the registry ------------------------------------------


@pytest.fixture()
def episode(tmp_path):
    """Audio whose timeline matches TURNS: Alice 0-4 and 10-12, Bob 5-9."""
    track = build_track([(ALICE, 5.0), (BOB, 5.0), (ALICE, 11.0)])
    wav = tmp_path / "ep.wav"
    sf.write(wav, track, SR)
    return wav


def test_audio_dry_run_shows_likely_identity(tmp_path, backend, episode):
    db = tmp_path / "reg.db"
    solo = tmp_path / "alice.wav"
    sf.write(solo, build_track([(ALICE, 6.0)]), SR)
    with Registry(db) as registry:
        enroll_from_audio("Alice", solo, registry, backend)

    with Registry(db) as registry:
        report = inspect_clusters(
            write_json(tmp_path / "ep.json"), audio=episode,
            registry=registry, backend=backend, match=FAKE,
        )

    by_label = {c.label: c for c in report.clusters}
    assert by_label["SPEAKER_00"].match.identity_name == "Alice"
    assert by_label["SPEAKER_00"].match.status == MATCHED
    assert by_label["SPEAKER_01"].match.status == NEW
    assert report.matched_against_registry

    summary = report.summary()
    assert "likely: Alice" in summary
    # the unmatched cluster must not be described as "likely" anyone
    assert "no match (closest: Alice" in summary
    assert summary.count("likely:") == 1

    # strictly read-only: no new identities, no assignments recorded
    with Registry(db) as registry:
        assert [i.name for i in registry.identities()] == ["Alice"]
        assert registry.conn.execute("SELECT COUNT(*) FROM assignments").fetchone()[0] == 0
        assert registry.conn.execute("SELECT COUNT(*) FROM voiceprints").fetchone()[0] == 1


def test_audio_with_empty_registry_says_so(tmp_path, backend, episode):
    with Registry(tmp_path / "reg.db") as registry:
        report = inspect_clusters(
            write_json(tmp_path / "ep.json"), audio=episode,
            registry=registry, backend=backend, match=FAKE,
        )
    assert all(c.match is None for c in report.clusters)
    assert report.registry_empty
    assert not report.matched_against_registry
    assert "no identities yet" in report.summary()


def test_without_audio_hints_at_it(tmp_path):
    report = inspect_clusters(write_json(tmp_path / "ep.json"))
    assert all(c.match is None for c in report.clusters)
    assert "Pass --audio" in report.summary()


# -- JSON shape ------------------------------------------------------------


def test_json_shape(tmp_path):
    data = inspect_clusters(write_json(tmp_path / "ep.json")).to_dict()

    assert data["totals"] == {"clusters": 2, "segments": 4, "speech_seconds": 11.0}
    assert data["audio"] is None
    top = data["clusters"][0]
    assert top["cluster"] == "SPEAKER_00"
    assert top["speech_seconds"] == 7.0
    assert top["segments"] == 3
    assert top["share"] == pytest.approx(0.6364, abs=1e-4)
    assert top["previews"]["longest"]["start"] == 0.0
    assert top["previews"]["middle"]["start"] == 10.0
    assert top["match"] is None
    assert data["clusters"][1]["previews"]["middle"] is None


# -- CLI -------------------------------------------------------------------


def test_cli_clusters_text_and_json(tmp_path, episode):
    path = write_json(tmp_path / "ep.json")
    db = str(tmp_path / "reg.db")

    result = runner.invoke(app, ["clusters", str(path), "--db", db])
    assert result.exit_code == 0, result.output
    assert "SPEAKER_00" in result.output
    assert "middle of the episode" in result.output
    assert "speakerdex enroll" in result.output  # tells you the next command

    result = runner.invoke(app, ["clusters", str(path), "--db", db, "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["totals"]["clusters"] == 2


def test_cli_clusters_with_audio(tmp_path, episode):
    db = str(tmp_path / "reg.db")
    solo = tmp_path / "alice.wav"
    sf.write(solo, build_track([(ALICE, 6.0)]), SR)
    assert runner.invoke(
        app, ["enroll", "Alice", str(solo), "--backend", "fake-spectral", "--db", db]
    ).exit_code == 0

    result = runner.invoke(
        app,
        ["clusters", str(write_json(tmp_path / "ep.json")), "--audio", str(episode),
         "--backend", "fake-spectral", "--db", db, "--json", *FAKE_FLAGS],
    )
    assert result.exit_code == 0, result.output
    by_label = {c["cluster"]: c for c in json.loads(result.output)["clusters"]}
    alice = by_label["SPEAKER_00"]["match"]
    assert alice["identity"] == "Alice"
    assert alice["status"] == MATCHED
    assert alice["similarity"] == pytest.approx(1.0, abs=0.01)
    assert by_label["SPEAKER_01"]["match"]["status"] == NEW
