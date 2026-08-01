"""Calibration on synthetic voices: distributions separate, thresholds sane."""

import json

import pytest
import soundfile as sf
from typer.testing import CliRunner

from speakerdex.calibrate import calibrate

from .conftest import SR, build_track

runner = CliRunner()

VOICES = {"alex": 220.0, "sam": 520.0, "jordan": 800.0}


@pytest.fixture()
def voices_dir(tmp_path):
    root = tmp_path / "voices"
    for name, f0 in VOICES.items():
        d = root / name
        d.mkdir(parents=True)
        for i in range(3):
            sf.write(d / f"clip{i}.wav", build_track([(f0, 3.0)], seed=i * 10), SR)
    return root


def test_calibrate_separates_and_recommends(voices_dir, backend):
    report = calibrate(voices_dir, backend)
    assert report.n_speakers == 3
    assert report.n_files == 9
    assert min(report.same) > max(report.diff)  # clean separation on synthetic voices
    assert report.review_threshold < report.match_threshold
    assert max(report.diff) < report.review_threshold < min(report.same)
    assert report.top1_accuracy == 1.0
    assert report.warnings == []
    text = report.summary()
    assert "--match-threshold" in text
    assert "100%" in text


def test_calibrate_needs_two_speakers(tmp_path, backend):
    root = tmp_path / "voices"
    d = root / "only_one"
    d.mkdir(parents=True)
    sf.write(d / "a.wav", build_track([(220.0, 2.0)]), SR)
    with pytest.raises(ValueError, match="at least 2 speaker"):
        calibrate(root, backend)


def test_calibrate_cli_json(voices_dir):
    from speakerdex.cli import app

    result = runner.invoke(
        app, ["calibrate", str(voices_dir), "--backend", "fake-spectral", "--json"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["top1_accuracy"] == 1.0
    assert data["review_threshold"] < data["match_threshold"]
