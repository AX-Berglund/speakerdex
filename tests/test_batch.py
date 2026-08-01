"""Batch processing over a folder of episodes, on synthetic voices.

The season below is the scenario batch mode exists for: two people recur across
three episodes, and the diarizer gives them *different* cluster labels in every
file (even swapping the numbering in ep02). One `process-dir --enroll-unknowns`
should collapse that into exactly two identities.

Note the thresholds: the fake spectral backend scores the same synthetic voice
at ~1.0 and two *different* ones at ~0.67 — a different scale from ECAPA, whose
0.55/0.40 defaults the CLI ships. Tests that put one voice in a file alone must
therefore use thresholds calibrated for this backend, or a lone stranger clears
0.55 and false-matches. (Files holding two voices hide that: the matcher's
greedy 1:1 rule lets only one cluster claim an identity.)
"""

import json

import pytest
import soundfile as sf
from typer.testing import CliRunner

from speakerdex.adapters.rttm import write_rttm
from speakerdex.batch import (
    ALREADY_PROCESSED,
    NO_DIARIZATION,
    PROCESSED,
    pair_files,
    process_dir,
)
from speakerdex.cli import app
from speakerdex.matcher import MatchConfig
from speakerdex.pipeline import ProcessConfig
from speakerdex.registry import Registry
from speakerdex.types import MATCHED, Segment

from .conftest import SR, build_track

runner = CliRunner()

ALICE, BOB = 220.0, 520.0
SPEECH_SEC = 4.0

# Calibrated for the fake spectral backend (same voice ~1.0, different ~0.67).
FAKE_MATCH, FAKE_REVIEW = 0.90, 0.80
FAKE_THRESHOLDS = [
    "--match-threshold", str(FAKE_MATCH),
    "--review-threshold", str(FAKE_REVIEW),
]


def write_episode(directory, stem, voices, seed=0, diarization=".rttm"):
    """Write <stem>.wav plus its diarization; voices is [(f0, cluster_label), ...].

    Each voice speaks SPEECH_SEC seconds, back to back, in the order given.
    """
    directory.mkdir(parents=True, exist_ok=True)
    sf.write(
        directory / f"{stem}.wav",
        build_track([(f0, SPEECH_SEC) for f0, _ in voices], seed=seed),
        SR,
    )
    segments = [
        Segment(i * SPEECH_SEC, (i + 1) * SPEECH_SEC, label)
        for i, (_, label) in enumerate(voices)
    ]
    if diarization == ".rttm":
        write_rttm(segments, directory / f"{stem}.rttm", file_id=stem)
    elif diarization == ".json":
        (directory / f"{stem}.json").write_text(
            json.dumps(
                {
                    "segments": [
                        {"start": s.start, "end": s.end, "speaker": s.speaker} for s in segments
                    ]
                }
            )
        )
    return directory / f"{stem}.wav"


@pytest.fixture()
def season(tmp_path):
    """Three episodes, two recurring voices, different cluster labels each time."""
    d = tmp_path / "season1"
    write_episode(d, "ep01", [(ALICE, "SPEAKER_00"), (BOB, "SPEAKER_01")], seed=0)
    write_episode(d, "ep02", [(BOB, "SPEAKER_00"), (ALICE, "SPEAKER_01")], seed=7)
    write_episode(d, "ep03", [(ALICE, "SPEAKER_05"), (BOB, "SPEAKER_09")], seed=13)
    return d


def run(directory, db, backend, force=False, enroll_unknowns=True):
    config = ProcessConfig(
        match=MatchConfig(match_threshold=FAKE_MATCH, review_threshold=FAKE_REVIEW),
        enroll_unknowns=enroll_unknowns,
    )
    with Registry(db) as registry:
        return process_dir(directory, registry, backend, config, force=force)


# -- pairing ---------------------------------------------------------------


def test_pairs_by_stem_and_reports_missing(tmp_path):
    d = tmp_path / "mixed"
    write_episode(d, "ep01", [(ALICE, "SPEAKER_00")], diarization=".rttm")
    write_episode(d, "ep02", [(BOB, "SPEAKER_00")], diarization=".json")
    write_episode(d, "ep03", [(ALICE, "SPEAKER_00")], diarization=None)  # orphan audio
    (d / "notes.txt").write_text("not audio")

    pairs = pair_files(d)
    assert [p.audio.name for p in pairs] == ["ep01.wav", "ep02.wav", "ep03.wav"]
    assert pairs[0].diarization.name == "ep01.rttm"
    assert pairs[1].diarization.name == "ep02.json"
    assert pairs[2].diarization is None


def test_rttm_wins_when_both_partners_exist(tmp_path):
    d = tmp_path / "both"
    write_episode(d, "ep01", [(ALICE, "SPEAKER_00")], diarization=".rttm")
    write_episode(d, "ep01", [(ALICE, "SPEAKER_00")], diarization=".json")
    assert pair_files(d)[0].diarization.name == "ep01.rttm"


def test_skips_scratch_entries_and_subdirs(tmp_path):
    d = tmp_path / "scratch"
    write_episode(d, "ep01", [(ALICE, "SPEAKER_00")])
    write_episode(d, "_cache", [(BOB, "SPEAKER_00")])
    write_episode(d, "._applefile", [(BOB, "SPEAKER_00")])
    write_episode(d / "nested", "ep99", [(BOB, "SPEAKER_00")])  # not recursive in v1
    assert [p.audio.name for p in pair_files(d)] == ["ep01.wav"]


def test_missing_diarization_is_skipped_not_fatal(tmp_path, backend):
    d = tmp_path / "orphan"
    write_episode(d, "ep01", [(ALICE, "SPEAKER_00")])
    write_episode(d, "ep02", [(BOB, "SPEAKER_00")], diarization=None)

    report = run(d, tmp_path / "reg.db", backend)
    statuses = {r.audio.name: r.status for r in report.results}
    assert statuses == {"ep01.wav": PROCESSED, "ep02.wav": NO_DIARIZATION}
    assert "no diarization file" in report.summary()


# -- the scenario batch mode exists for -------------------------------------


def test_recurring_voices_collapse_to_two_identities(season, tmp_path, backend):
    """Definition of done: 3 episodes, 2 voices, 2 identities, later files matched."""
    db = tmp_path / "reg.db"
    report = run(season, db, backend)

    assert [r.status for r in report.results] == [PROCESSED] * 3

    with Registry(db) as registry:
        identities = registry.identities()
    assert len(identities) == 2, [i.name for i in identities]

    # ep01 births both identities; every later cluster matches an existing one
    later = [d for r in report.results[1:] for d in r.decisions]
    assert len(later) == 4
    assert all(d.status == MATCHED for d in later), [str(d) for d in later]

    # each identity appears in all three episodes, under whatever label
    roster = report.roster()
    assert sorted(roster) == ["Unknown-1", "Unknown-2"]
    assert all(files == ["ep01.wav", "ep02.wav", "ep03.wav"] for files in roster.values())
    assert report.cluster_counts() == {"matched": 4, "review": 0, "new": 2}


def test_ordering_decides_which_file_an_identity_is_born_in(tmp_path, backend):
    """Unknown-N numbering follows sorted filename order, not disk order."""
    d = tmp_path / "ordered"
    # ep02 written first, and Bob leads it — but ep01 sorts first, so Alice is Unknown-1
    write_episode(d, "ep02", [(BOB, "SPEAKER_00")], seed=7)
    write_episode(d, "ep01", [(ALICE, "SPEAKER_00")], seed=0)

    report = run(d, tmp_path / "reg.db", backend)
    assert [r.audio.name for r in report.results] == ["ep01.wav", "ep02.wav"]
    assert report.roster() == {"Unknown-1": ["ep01.wav"], "Unknown-2": ["ep02.wav"]}


# -- idempotency -----------------------------------------------------------


def test_rerun_is_idempotent(season, tmp_path, backend):
    db = tmp_path / "reg.db"
    run(season, db, backend)

    with Registry(db) as registry:
        before = registry.conn.execute(
            "SELECT source, cluster, identity_id, status FROM assignments ORDER BY source, cluster"
        ).fetchall()
        voiceprints_before = registry.conn.execute("SELECT COUNT(*) FROM voiceprints").fetchone()

    again = run(season, db, backend)

    assert [r.status for r in again.results] == [ALREADY_PROCESSED] * 3
    assert len(again.processed) == 0
    assert "already processed" in again.summary()
    # skipped files still report their stored decisions, so the roster survives
    assert again.roster() == {
        "Unknown-1": ["ep01.wav", "ep02.wav", "ep03.wav"],
        "Unknown-2": ["ep01.wav", "ep02.wav", "ep03.wav"],
    }

    with Registry(db) as registry:
        assert len(registry.identities()) == 2
        assert (
            registry.conn.execute(
                "SELECT source, cluster, identity_id, status FROM assignments "
                "ORDER BY source, cluster"
            ).fetchall()
            == before
        )
        assert registry.conn.execute("SELECT COUNT(*) FROM voiceprints").fetchone() == (
            voiceprints_before
        )


def test_force_reprocesses_without_duplicating(season, tmp_path, backend):
    db = tmp_path / "reg.db"
    run(season, db, backend)

    forced = run(season, db, backend, force=True)
    assert [r.status for r in forced.results] == [PROCESSED] * 3

    with Registry(db) as registry:
        # still two identities, and assignments upserted rather than appended
        assert len(registry.identities()) == 2
        assert registry.conn.execute("SELECT COUNT(*) FROM assignments").fetchone()[0] == 6
        assert len(registry.assignments_for(str(season / "ep01.wav"))) == 2


def test_new_episode_added_later_only_costs_the_new_file(season, tmp_path, backend):
    db = tmp_path / "reg.db"
    run(season, db, backend)
    write_episode(season, "ep04", [(BOB, "SPEAKER_02"), (ALICE, "SPEAKER_03")], seed=21)

    report = run(season, db, backend)
    statuses = {r.audio.name: r.status for r in report.results}
    assert statuses["ep04.wav"] == PROCESSED
    assert all(statuses[f"ep0{i}.wav"] == ALREADY_PROCESSED for i in (1, 2, 3))
    assert all(d.status == MATCHED for d in report.results[-1].decisions)

    with Registry(db) as registry:
        assert len(registry.identities()) == 2


# -- CLI -------------------------------------------------------------------


def test_cli_process_dir_text_and_json(season, tmp_path):
    db = str(tmp_path / "reg.db")
    args = [
        "process-dir", str(season),
        "--backend", "fake-spectral",
        "--db", db,
        *FAKE_THRESHOLDS,
    ]

    result = runner.invoke(app, [*args, "--enroll-unknowns"])
    assert result.exit_code == 0, result.output
    assert "ep01.wav" in result.output
    assert "Files: 3 processed, 0 skipped" in result.output
    assert "Identities seen:" in result.output

    # second run: everything skipped, reported as one JSON object
    result = runner.invoke(app, [*args, "--enroll-unknowns", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["summary"]["processed"] == 0
    assert data["summary"]["already_processed"] == 3
    assert data["summary"]["clusters"] == {"matched": 4, "review": 0, "new": 2}
    assert sorted(data["roster"]) == ["Unknown-1", "Unknown-2"]
    assert [f["status"] for f in data["files"]] == [ALREADY_PROCESSED] * 3
    assert data["files"][0]["decisions"][0]["speech_seconds"] is None  # unknown when replayed

    result = runner.invoke(app, [*args, "--force", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["summary"]["processed"] == 3
    assert data["files"][0]["decisions"][0]["speech_seconds"] == SPEECH_SEC


def test_empty_registry_similarity_reads_na_and_serializes_null(tmp_path):
    """The first file against a fresh registry has nothing to compare against."""
    d = tmp_path / "fresh"
    write_episode(d, "ep01", [(ALICE, "SPEAKER_00")])
    def args(db):  # each invocation needs a fresh registry to still be empty
        return [
            "process-dir", str(d),
            "--backend", "fake-spectral",
            "--db", str(tmp_path / db),
            "--enroll-unknowns",
            *FAKE_THRESHOLDS,
        ]

    result = runner.invoke(app, args("text.db"))
    assert result.exit_code == 0, result.output
    assert "sim=n/a" in result.output
    assert "-1.000" not in result.output

    result = runner.invoke(app, [*args("json.db"), "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["files"][0]["decisions"][0]["similarity"] is None


def test_cli_reports_empty_directory(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    result = runner.invoke(
        app, ["process-dir", str(d), "--backend", "fake-spectral", "--db", str(tmp_path / "r.db")]
    )
    assert result.exit_code == 0, result.output
    assert "No audio files found" in result.output
