# First real-world run

The first end-to-end test of speakerdex on real conversational audio, following
[real-world-testing.md](real-world-testing.md) Step 2. Everything below is
measured, not estimated.

**Result: PASS.** The recurring host resolved to the same identity in all three
episodes at `matched` confidence, including one episode where the diarizer gave
him a different cluster label. Zero misclassifications, zero review-band traffic.

## Setup

| | |
|---|---|
| Show | [The Changelog](https://changelog.com/podcast) — *Interview* format |
| Episodes | #681 *MCP on Code Mode*, #682 *From open source hits to OpenAI*, #683 *Canary tokens and digital tripwires* |
| Material | 12 min excerpt per episode from t=90s, 16 kHz mono, stream-trimmed with ffmpeg |
| Transcription | whisperx 3.8.6, model `tiny`, `--language en`, `--device cpu`, `--compute_type int8`, `--threads 8` |
| Diarization | `pyannote/speaker-diarization-community-1` (whisperx default), pyannote.audio 4.0.7 |
| Matching | speakerdex 0.1.0, `ecapa` backend, torch 2.8.0, speechbrain 1.1.0 |
| Hardware | Apple Silicon, 8 cores, CPU only |

whisperx installed into the same virtualenv as speakerdex without conflict —
`pip check` clean. It did pull torch down from 2.13.0 to 2.8.0; the ECAPA
backend and the full test suite still pass on that version. The separate
throwaway venv the plan allowed for was not needed.

Diarization wall clock, for 12 minutes of audio each: **ep01 ~14 min, ep02
10m04s, ep03 9m35s** — roughly 0.8–1.2× realtime. This is why the episodes were
trimmed; the full 1.5–2 hour episodes would have run for hours.

### Ground truth

Established independently of speakerdex, from the feed's `podcast:person` tags
and by reading the transcript of each cluster:

| Episode | Host | Guest | Third voice |
|---|---|---|---|
| ep01 (#681) | Adam Stacoviak — `SPEAKER_01` | Matt Carey — `SPEAKER_02` | Coder.com ad read — `SPEAKER_00` |
| ep02 (#682) | Adam Stacoviak — `SPEAKER_01` | Max Stoiber — `SPEAKER_02` | Coder.com ad read — `SPEAKER_00` |
| ep03 (#683) | Adam Stacoviak — **`SPEAKER_00`** | Haroon Meer — `SPEAKER_01` | — |

Two things worth noting before the results. The Changelog's Interview format is
**single-hosted**, so this is one recurring voice across three files rather than
the two the plan assumed — and the three distinct guests become genuine negative
controls, which is arguably the better test. And the host's cluster label is
`SPEAKER_01` in two episodes but `SPEAKER_00` in the third: exactly the
per-file label permutation speakerdex exists to undo.

Diarization quality was good: 152/160/132 segments per episode, **zero
unlabelled**, 618/621/644 seconds of labelled speech. One error spotted — in
ep03 at 01:31 a host interjection is attributed to the guest's cluster.

## Commands

```bash
whisperx episodes/ep01.mp3 --model tiny --language en --diarize \
    --output_dir episodes --output_format json \
    --device cpu --compute_type int8 --threads 8      # ×3

speakerdex init
speakerdex enroll "Adam Stacoviak" episodes/ep01.mp3 \
    --diarization episodes/ep01.json --cluster SPEAKER_01

speakerdex process-dir episodes/ --enroll-unknowns    # defaults: 0.55 / 0.40
```

## Results

```
Files: 3 processed, 0 skipped
Clusters: 4 matched, 0 review, 4 new
Identities seen:
  Adam Stacoviak  ep01.mp3, ep02.mp3, ep03.mp3
  Unknown-1       ep01.mp3, ep02.mp3
  Unknown-2       ep01.mp3
  Unknown-3       ep02.mp3
  Unknown-4       ep03.mp3
```

Every cluster, against ground truth:

| File | Cluster | Actually | speakerdex | Status | Similarity | Speech |
|---|---|---|---|---|---|---|
| ep01 | `SPEAKER_01` | Adam Stacoviak | Adam Stacoviak | matched | 1.000 † | 65.2s |
| ep01 | `SPEAKER_00` | Coder ad read | Unknown-1 | new | 0.143 | 61.7s |
| ep01 | `SPEAKER_02` | Matt Carey | Unknown-2 | new | 0.051 | 66.8s |
| ep02 | `SPEAKER_01` | Adam Stacoviak | **Adam Stacoviak** | **matched** | **0.916** | 66.7s |
| ep02 | `SPEAKER_00` | Coder ad read | Unknown-1 | matched | 0.997 ‡ | 62.0s |
| ep02 | `SPEAKER_02` | Max Stoiber | Unknown-3 | new | 0.221 | 63.8s |
| ep03 | `SPEAKER_00` | Adam Stacoviak | **Adam Stacoviak** | **matched** | **0.909** | 61.1s |
| ep03 | `SPEAKER_01` | Haroon Meer | Unknown-4 | new | 0.288 | 62.2s |

† ep01's host cluster is the audio the voiceprint was enrolled from, so 1.000 is
tautological and is excluded from all threshold analysis below.

‡ The Coder ad is a pre-roll baked into both episodes at nearly the same offset,
with near-identical transcript text. 0.997 reflects **near-duplicate audio**, not
cross-recording voice generalization, and is likewise excluded from the analysis.

**Every one of the eight clusters is classified correctly.** Five identities
exist at the end, which is exactly the five distinct voices in the material.
The two genuine cross-file identity matches are the host at **0.916** and
**0.909**. Nothing landed in the review band, so `speakerdex review` is empty
and no threshold retuning was required.

Re-running `process-dir` changed nothing — 3 files skipped as already
processed, same roster, same 5 identities.

### Reading the `new` outcomes

All four are correct. Three are the distinct guests, who each appear in exactly
one episode and should never match anything. The fourth, ep01's `SPEAKER_00`, is
the sponsor voice on first sight; it correctly became a new identity and then
correctly matched itself in ep02.

The sponsor case is a legitimate finding for podcast use rather than a bug:
baked-in ad reads are recurring voices, so speakerdex faithfully tracks them as
people. On a real catalogue this would accumulate identities nobody asked for.

## Threshold observations

Excluding the two artifacts marked above:

| | |
|---|---|
| Lowest true positive | **0.909** (host, ep03) |
| Highest true negative | **0.288** (Haroon Meer vs. his nearest of 4 enrolled identities) |
| Separation gap | **0.621** |

Any `--match-threshold` in **(0.288, 0.909]** classifies this material perfectly.
The shipped defaults sit near the middle of that window with wide margins on
both sides — 0.262 of headroom above the highest impostor, 0.359 below the
weakest true match.

A sweep on a copy of the registry confirms the upper edge: at
`--match-threshold 0.92`, both genuine host matches (0.916, 0.909) drop into the
review band. One caveat on that sweep — after the first run the guests are
themselves enrolled, so a `--force` re-run matches them to themselves at ~1.0.
It is therefore only informative about the host boundary; the true-negative
numbers must come from the first run, where each guest was scored against a
registry that did not yet contain it.

### Versus the LibriVox calibration

| | LibriVox read speech | This podcast |
|---|---|---|
| Same speaker | 0.729 – 0.926 (p5 0.756) | 0.909 – 0.916 |
| Different speaker | 0.049 – 0.249 (p99 0.246) | 0.051 – 0.288 |
| Separation | 0.510 | 0.621 |
| Recommended / used | 0.55 / 0.37 | 0.55 / 0.40 defaults, unchanged |

The interesting result is that **conversational podcast audio scored *better*
than read audiobook speech**, which inverts the expectation in
real-world-testing.md that read speech is a best case.

The likely reason is channel consistency, not speech style: it is the same
person on the same microphone through the same studio chain in all three
episodes, whereas the LibriVox narrators' chapters may span recording sessions
and setups. That makes this a *favourable* real-world case rather than a
representative one. A host recorded remotely on varying equipment, or a guest
appearing months apart, should be expected to score lower. Each cluster here
also aggregated ~60s of clean speech; short clusters will be noisier, which is
what the review band is for.

The defaults need no change on this evidence.

## Caveats

- 12-minute excerpts, not full episodes.
- Only **two** genuine cross-file identity matches. Small n; this is one data
  point, not a benchmark.
- One voiceprint per identity — `--reinforce` was not used, so every centroid is
  a single sample.
- `tiny` was chosen for speed and its transcripts are rough. Fine for segment
  boundaries, but it garbled the names that manual attribution relies on
  ("Matt, carry" for Matt Carey), which made identifying clusters harder.
- Trimming from t=90s cut the show intro, where hosts self-identify — the
  easiest signal for working out which cluster is whom. Start excerpts at t=0.

## CLI friction

Collected while actually doing the above. Ordered by how much time each cost.

1. **No way to inspect the clusters in a diarization file.** `enroll --cluster
   SPEAKER_XX` demands a cluster ID, but nothing in speakerdex will tell you
   what clusters exist, how long each speaks, or what they said. Writing a
   throwaway script to dump per-cluster speaking time plus sample transcript
   text was the single largest piece of manual work in this session, and every
   user doing step 4 will write the same script. A `speakerdex clusters
   <diarization>` command would remove it.
2. **"Longest speaking time = the host" is wrong on interview shows.** The guest
   outspoke the host 58% to 24% in ep01. The heuristic in the testing guide
   picks the wrong cluster on exactly the format most people will reach for.
3. **`Unknown-N` is opaque.** The roster says which files an identity appears in
   but nothing about who it is, so naming means going back to the transcript
   with timestamps in hand. Storing one representative `(source, start, end)`
   per identity would make `ls` actionable.
4. **Speech seconds are missing from text output.** How much audio backed a
   match is central to trusting it, but it only appears in `--json`.
5. **`enroll` doesn't report how much speech it used.** A cluster with three
   seconds of audio silently produces a weak voiceprint that poisons later
   matching. It should print the seconds, and probably warn below a threshold.
6. **No way to exclude a time range.** Baked-in sponsor reads become permanent
   identities, and on a full catalogue they would pile up.
7. **`ls` and the roster don't overlap.** `ls` gives voiceprint counts, the
   roster gives file presence; neither gives both.
8. **Nothing reports your observed margins.** After a run, speakerdex knows the
   similarity of every decision and could say "your closest call was 0.29 below
   the match threshold" — the number that tells you whether your thresholds are
   safe. Today that requires post-processing `--json` by hand.
