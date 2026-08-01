# First real-world test

The unit tests prove the *mechanics* work (enroll in file 1, recognize in file 2). What they can't prove is how ECAPA similarity scores distribute on **real voices in your material** — which is what the default thresholds (`match ≥ 0.55`, `review ≥ 0.40`) depend on. This guide is the ~30-minute session that answers that.

> Note: this needs the real embedding model, so run it on your own machine:
> `pip install -e ".[ecapa]"` (first run downloads ~80 MB from Hugging Face).

## Step 0 — No material yet? Use the bundled starter set

If you don't have suitable recordings on hand, the repo ships a script that
builds a ready-to-calibrate `voices/` folder from public-domain LibriVox
audiobooks — three verified solo narrators (Elizabeth Klett, Moira Fogarty,
Stewart Wills), three one-minute clips each:

```bash
python scripts/setup_voices.py     # downloads ~70 MB once, caches locally
speakerdex calibrate voices/
```

Read speech is cleaner than conversation, so treat the resulting thresholds
as a best case; recalibrate on your real material (below) when you have it.

## Step 1 — Calibrate on voices you know

Pick 3–6 people you have multiple recordings of (podcast hosts are perfect: grab 3+ episodes). Cut a clean solo clip per person per episode — 30–60 seconds where only they speak:

```bash
# example: cut 0:45–1:30 out of an episode as a solo clip
ffmpeg -i ep01.mp3 -ss 45 -to 90 -ac 1 -ar 16000 voices/alex/ep01.wav
```

Layout:

```
voices/
  alex/    ep01.wav  ep05.wav  ep09.wav
  sam/     ep01.wav  ep03.wav
  jordan/  ep02.wav  ep07.wav
```

Then:

```bash
speakerdex calibrate voices/
```

You get the same/different-speaker similarity distributions, an ASCII histogram, leave-one-out identification accuracy, and **recommended `--match-threshold` / `--review-threshold` values measured on your audio**. What to look for:

- **Clean separation** (same-speaker p5 well above diff-speaker p99): trust the recommended thresholds.
- **Overlap warning**: usually means enrollment clips are too short, noisy, or contain music/other voices. Re-cut cleaner clips before touching thresholds.
- **Leave-one-out below ~90%**: same advice — fix the data first, then consider more voiceprints per person (`--reinforce` later helps here).

## Step 2 — End-to-end on two episodes

Produce diarization for two episodes with whichever tool you prefer.

WhisperX (transcript + diarization in one JSON):

```bash
whisperx ep01.mp3 --diarize --hf_token $HF_TOKEN --output_format json
```

Or pyannote to RTTM:

```python
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
with open("ep01.rttm", "w") as f:
    pipeline("ep01.wav").write_rttm(f)
```

Then the speakerdex loop, using your calibrated thresholds:

```bash
speakerdex init
speakerdex enroll "Alex" voices/alex/ep01.wav
speakerdex enroll "Sam"  voices/sam/ep01.wav

speakerdex process ep01.mp3 -d ep01.json --enroll-unknowns \
    --match-threshold 0.58 --review-threshold 0.42

speakerdex process ep02.mp3 -d ep02.json --enroll-unknowns \
    --match-threshold 0.58 --review-threshold 0.42
```

If both episodes and their diarization files sit in one folder with matching
stems (`ep01.mp3` + `ep01.json`, `ep02.mp3` + `ep02.json`), the two `process`
calls above collapse into one:

```bash
speakerdex process-dir episodes/ --enroll-unknowns \
    --match-threshold 0.58 --review-threshold 0.42
```

Same result, plus a roster at the end showing which identity turned up in which
episode — which is the answer to the question this step asks, without diffing
two blocks of output by eye:

```
Files: 2 processed, 0 skipped
Clusters: 3 matched, 0 review, 1 new
Identities seen:
  Alex       ep01.mp3, ep02.mp3
  Sam        ep01.mp3, ep02.mp3
  Unknown-1  ep02.mp3
```

Because files are taken in sorted filename order, `ep01` is where a recurring
voice gets enrolled and `ep02` is where it has to be recognized — the direction
you want for this test. Re-running is free: already-processed files are skipped
unless you pass `--force`, so you can add `ep03` later and only pay for it.

**The test that matters**: does the host resolve to the same name in both episodes, at `matched` confidence? Then:

```bash
speakerdex review                      # borderline calls, if any
speakerdex confirm ep02.mp3 SPEAKER_03 "Sam"
speakerdex ls
speakerdex process ep02.mp3 -d ep02.json -o ep02_named.json   # named transcript
```

## Step 3 — Record what you find

Open an issue (or note in the README) with: number of speakers, audio conditions, the calibrate output block, and whether the defaults (0.55/0.40) were close. After a few corpora this becomes the documented per-domain threshold guidance — and the input to the roadmap's automatic calibration.

## Known limitations to watch for

- **Overlapping speech**: segments where the diarizer interleaves speakers contaminate cluster embeddings. If a mostly-wrong match appears on a talk-over-heavy show, that's likely why (overlap-aware exclusion is on the roadmap).
- **Very short clusters**: someone who speaks < ~5 seconds in an episode gets a noisy embedding; expect `review` rather than `matched` for them, which is the intended behavior.
- **Music/jingles diarized as speech**: enroll nothing from intros; consider trimming them before processing.
