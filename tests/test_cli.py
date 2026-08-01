"""CLI smoke tests using the fake spectral backend."""

import json

from typer.testing import CliRunner

from speakerdex.adapters.rttm import write_rttm
from speakerdex.cli import app
from speakerdex.types import Segment

from .conftest import build_track

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip()


def test_init(tmp_path):
    db = tmp_path / "reg.db"
    result = runner.invoke(app, ["init", "--db", str(db)])
    assert result.exit_code == 0
    assert db.exists()


def test_full_cli_flow(tmp_path, write_wav):
    db = str(tmp_path / "reg.db")

    # enroll Alice from a solo clip
    alice_wav = write_wav("alice.wav", build_track([(220.0, 5.0)]))
    result = runner.invoke(
        app, ["enroll", "Alice", str(alice_wav), "--backend", "fake-spectral", "--db", db]
    )
    assert result.exit_code == 0, result.output

    # process an episode where Alice speaks first, a stranger second
    ep_wav = write_wav("ep1.wav", build_track([(220.0, 4.0), (520.0, 4.0)], seed=5))
    rttm = tmp_path / "ep1.rttm"
    write_rttm(
        [Segment(0.0, 4.0, "SPEAKER_00"), Segment(4.0, 8.0, "SPEAKER_01")], rttm, file_id="ep1"
    )
    result = runner.invoke(
        app,
        [
            "process", str(ep_wav),
            "--diarization", str(rttm),
            "--backend", "fake-spectral",
            "--enroll-unknowns",
            "--json",
            "--db", db,
        ],
    )
    assert result.exit_code == 0, result.output
    decisions = {d["cluster"]: d for d in json.loads(result.output)}
    assert decisions["SPEAKER_00"]["identity"] == "Alice"
    assert decisions["SPEAKER_00"]["status"] == "matched"
    assert decisions["SPEAKER_01"]["identity"] == "Unknown-1"

    # rename the stranger once we know who it is
    result = runner.invoke(app, ["rename", "Unknown-1", "Bob", "--db", db])
    assert result.exit_code == 0

    result = runner.invoke(app, ["ls", "--db", db])
    assert result.exit_code == 0
    assert "Alice" in result.output
    assert "Bob" in result.output


def test_enroll_wrong_cluster_fails_with_a_recoverable_listing(tmp_path, write_wav):
    """A typo'd cluster must be fixable from the error text alone."""
    db = str(tmp_path / "reg.db")
    wav = write_wav("ep.wav", build_track([(220.0, 5.0), (520.0, 3.0)]))
    rttm = tmp_path / "ep.rttm"
    write_rttm(
        [Segment(0.0, 5.0, "SPEAKER_00"), Segment(5.0, 8.0, "SPEAKER_01")], rttm, file_id="ep"
    )

    result = runner.invoke(
        app,
        ["enroll", "Bob", str(wav), "-d", str(rttm), "--cluster", "SPEAKER_99",
         "--backend", "fake-spectral", "--db", db],
    )
    assert result.exit_code == 1
    combined = result.output + str(result.exception or "")
    assert "SPEAKER_00" in combined and "5.0s" in combined
    assert "Traceback" not in combined  # a user error, not a crash

    # the listing is enough to retry successfully
    assert runner.invoke(
        app,
        ["enroll", "Bob", str(wav), "-d", str(rttm), "--cluster", "SPEAKER_01",
         "--backend", "fake-spectral", "--db", db],
    ).exit_code == 0


def test_enroll_reports_speech_seconds_and_warns_when_thin(tmp_path, write_wav):
    db = str(tmp_path / "reg.db")
    thin = write_wav("thin.wav", build_track([(220.0, 3.0)]))
    result = runner.invoke(
        app, ["enroll", "Alice", str(thin), "--backend", "fake-spectral", "--db", db]
    )
    assert result.exit_code == 0, result.output
    assert "3.0s of speech" in result.output
    assert "Warning" in result.output and "unreliable" in result.output

    plenty = write_wav("plenty.wav", build_track([(520.0, 20.0)]))
    result = runner.invoke(
        app, ["enroll", "Bob", str(plenty), "--backend", "fake-spectral", "--db", db]
    )
    assert result.exit_code == 0, result.output
    assert "20.0s of speech" in result.output
    assert "Warning" not in result.output


def test_ls_shows_file_presence(tmp_path, write_wav):
    db = str(tmp_path / "reg.db")
    wav = write_wav("alice.wav", build_track([(220.0, 6.0)]))
    runner.invoke(app, ["enroll", "Alice", str(wav), "--backend", "fake-spectral", "--db", db])

    result = runner.invoke(app, ["ls", "--db", db])
    assert "seen in 0 file(s)" in result.output  # enrolled, not yet processed

    rttm = tmp_path / "alice.rttm"
    write_rttm([Segment(0.0, 6.0, "SPEAKER_00")], rttm, file_id="alice")
    runner.invoke(
        app, ["process", str(wav), "-d", str(rttm), "--backend", "fake-spectral", "--db", db]
    )
    result = runner.invoke(app, ["ls", "--db", db])
    assert "seen in 1 file(s)" in result.output


def test_process_against_empty_registry_reports_na_not_minus_one(tmp_path, write_wav):
    """Nothing enrolled yet: there is no similarity to report, in text or JSON."""
    db = str(tmp_path / "reg.db")
    ep_wav = write_wav("ep1.wav", build_track([(220.0, 4.0)]))
    rttm = tmp_path / "ep1.rttm"
    write_rttm([Segment(0.0, 4.0, "SPEAKER_00")], rttm, file_id="ep1")
    args = ["process", str(ep_wav), "-d", str(rttm), "--backend", "fake-spectral", "--db", db]

    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "sim=n/a" in result.output
    assert "-1.000" not in result.output

    result = runner.invoke(app, [*args, "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)[0]["similarity"] is None
