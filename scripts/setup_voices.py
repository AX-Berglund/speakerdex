#!/usr/bin/env python
"""Build a ready-to-calibrate voices/ folder from public-domain LibriVox audio.

Three solo audiobook narrators (one voice per book, identity known and
verified), three chapters each -> voices/<reader>/<chapter>.wav, 60 seconds
per clip at 16 kHz mono. Public domain, so redistribution and analysis are
unencumbered.

Usage (from the repo root, with the package installed):

    python scripts/setup_voices.py            # creates ./voices/
    speakerdex calibrate voices/

Requires network access (downloads ~70 MB of MP3s on first run; cached in
voices/_downloads/, so re-runs are free).
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import soundfile as sf

from speakerdex.audio import TARGET_SR, load_audio

BASE = "https://archive.org/download"

# Verified solo recordings: every file within a reader is the same voice.
VOICES: dict[str, list[str]] = {
    "elizabeth_klett": [  # Pride and Prejudice (LibriVox, solo)
        f"{BASE}/prideandprejudice_1005_librivox/prideandprejudice_01_austen_64kb.mp3",
        f"{BASE}/prideandprejudice_1005_librivox/prideandprejudice_02_austen_64kb.mp3",
        f"{BASE}/prideandprejudice_1005_librivox/prideandprejudice_03_austen_64kb.mp3",
    ],
    "moira_fogarty": [  # The Art of War (LibriVox, solo)
        f"{BASE}/art_of_war_librivox/art_of_war_01-02_sun_tzu_64kb.mp3",
        f"{BASE}/art_of_war_librivox/art_of_war_03-04_sun_tzu_64kb.mp3",
        f"{BASE}/art_of_war_librivox/art_of_war_05-06_sun_tzu_64kb.mp3",
    ],
    "stewart_wills": [  # Moby Dick (LibriVox, solo)
        f"{BASE}/moby_dick_librivox/mobydick_000_melville_64kb.mp3",
        f"{BASE}/moby_dick_librivox/mobydick_001_002_melville_64kb.mp3",
        f"{BASE}/moby_dick_librivox/mobydick_003_melville_64kb.mp3",
    ],
}


def download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached   {dest.name}")
        return dest
    print(f"  fetching {dest.name} ...", flush=True)
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(1 << 16):
            f.write(chunk)
    tmp.rename(dest)
    return dest


def make_clip(mp3: Path, wav: Path, offset: float, seconds: float) -> None:
    if wav.exists():
        print(f"  exists   {wav.name}")
        return
    wave, sr = load_audio(mp3)  # mono float32 @ 16 kHz
    a, b = int(offset * sr), int((offset + seconds) * sr)
    if len(wave) <= a:
        raise SystemExit(f"{mp3.name} is shorter than the requested offset ({offset}s)")
    sf.write(wav, wave[a:b], sr)
    print(f"  wrote    {wav.name} ({(min(b, len(wave)) - a) / sr:.0f}s @ {TARGET_SR} Hz)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="voices", help="Output folder (default: voices/)")
    parser.add_argument("--offset", type=float, default=20.0,
                        help="Seconds to skip at the start of each chapter (default: 20)")
    parser.add_argument("--seconds", type=float, default=60.0,
                        help="Clip length in seconds (default: 60)")
    args = parser.parse_args()

    out = Path(args.out)
    cache = out / "_downloads"
    cache.mkdir(parents=True, exist_ok=True)

    for reader, urls in VOICES.items():
        print(f"{reader}:")
        (out / reader).mkdir(exist_ok=True)
        for url in urls:
            mp3 = download(url, cache / url.rsplit("/", 1)[1])
            wav = out / reader / (mp3.stem.replace("_64kb", "") + ".wav")
            make_clip(mp3, wav, args.offset, args.seconds)

    print("\nDone. Next:\n  speakerdex calibrate", out)


if __name__ == "__main__":
    sys.exit(main())
