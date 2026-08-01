import json

import pytest

from speakerdex.adapters import load_segments, write_rttm, write_whisperx
from speakerdex.adapters.rttm import load_rttm
from speakerdex.adapters.whisperx import load_whisperx
from speakerdex.types import Segment

RTTM = """\
SPEAKER ep1 1 0.500 2.300 <NA> <NA> SPEAKER_00 <NA> <NA>
SPEAKER ep1 1 3.100 1.000 <NA> <NA> SPEAKER_01 <NA> <NA>
"""

WHISPERX = {
    "segments": [
        {"start": 0.5, "end": 2.8, "text": "hello", "speaker": "SPEAKER_00",
         "words": [{"word": "hello", "start": 0.5, "end": 0.9, "speaker": "SPEAKER_00"}]},
        {"start": 3.1, "end": 4.1, "text": "hi", "speaker": "SPEAKER_01"},
        {"start": 5.0, "end": 5.5, "text": "(music)"},  # no speaker -> skipped
    ]
}


def test_load_rttm(tmp_path):
    path = tmp_path / "ep1.rttm"
    path.write_text(RTTM)
    segments = load_rttm(path)
    assert segments == [
        Segment(0.5, 2.8, "SPEAKER_00"),
        Segment(3.1, 4.1, "SPEAKER_01"),
    ]


def test_rttm_round_trip_with_mapping(tmp_path):
    path = tmp_path / "ep1.rttm"
    path.write_text(RTTM)
    segments = load_rttm(path)
    out = tmp_path / "named.rttm"
    write_rttm(segments, out, file_id="ep1", mapping={"SPEAKER_00": "Alice Smith"})
    reloaded = load_rttm(out)
    assert reloaded[0].speaker == "Alice_Smith"
    assert reloaded[1].speaker == "SPEAKER_01"
    assert reloaded[0].start == pytest.approx(0.5)


def test_malformed_rttm_raises(tmp_path):
    path = tmp_path / "bad.rttm"
    path.write_text("SPEAKER ep1 1 0.5\n")
    with pytest.raises(ValueError, match="malformed"):
        load_rttm(path)


def test_load_whisperx(tmp_path):
    path = tmp_path / "ep1.json"
    path.write_text(json.dumps(WHISPERX))
    segments = load_whisperx(path)
    assert [s.speaker for s in segments] == ["SPEAKER_00", "SPEAKER_01"]


def test_write_whisperx_relabels_and_preserves(tmp_path):
    src = tmp_path / "ep1.json"
    src.write_text(json.dumps(WHISPERX))
    out = tmp_path / "named.json"
    write_whisperx(src, out, {"SPEAKER_00": "Alice"})
    data = json.loads(out.read_text())
    assert data["segments"][0]["speaker"] == "Alice"
    assert data["segments"][0]["speaker_cluster"] == "SPEAKER_00"
    assert data["segments"][0]["words"][0]["speaker"] == "Alice"
    assert data["segments"][1]["speaker"] == "SPEAKER_01"  # unmapped: untouched


def test_load_segments_infers_format(tmp_path):
    rttm = tmp_path / "a.rttm"
    rttm.write_text(RTTM)
    assert len(load_segments(rttm)) == 2
    unknown = tmp_path / "a.xyz"
    unknown.write_text("")
    with pytest.raises(ValueError, match="cannot infer"):
        load_segments(unknown)
