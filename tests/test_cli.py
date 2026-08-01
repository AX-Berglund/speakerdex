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
