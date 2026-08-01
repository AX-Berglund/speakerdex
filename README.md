# speakerdex

**The same voice gets the same name — across every file.**

Every diarization tool labels speakers `SPEAKER_00`, `SPEAKER_01`… and forgets
them when the file ends. Process episode 2 and your host is suddenly
`SPEAKER_03`. speakerdex is a local voice registry that sits on top of any
diarizer and resolves per-file speaker clusters to persistent, named identities.

```
$ speakerdex process-dir episodes/ --enroll-unknowns
ep01.mp3
  SPEAKER_00 -> Unknown-1 (new, sim=0.143, 62s)
  SPEAKER_01 -> Adam Stacoviak (matched, sim=1.000, 65s)
  SPEAKER_02 -> Unknown-2 (new, sim=0.051, 67s)
ep02.mp3
  SPEAKER_00 -> Unknown-1 (matched, sim=0.997, 62s)
  SPEAKER_01 -> Adam Stacoviak (matched, sim=0.916, 67s)
  SPEAKER_02 -> Unknown-3 (new, sim=0.221, 64s)
ep03.mp3
  SPEAKER_00 -> Adam Stacoviak (matched, sim=0.909, 61s)
  SPEAKER_01 -> Unknown-4 (new, sim=0.288, 62s)

Files: 3 processed, 0 skipped
Clusters: 4 matched, 0 review, 4 new
Identities seen:
  Adam Stacoviak  ep01.mp3, ep02.mp3, ep03.mp3
  Unknown-1       ep01.mp3, ep02.mp3
  Unknown-2       ep01.mp3
  Unknown-3       ep02.mp3
  Unknown-4       ep03.mp3
```

That is real output from the run in [docs/first-real-run.md](docs/first-real-run.md).
Note `ep03`: the diarizer gave the host `SPEAKER_00` there and `SPEAKER_01` in
the other two. The label changed; the identity didn't.

- **Bring your own diarizer** — consumes RTTM or WhisperX JSON; works with
  pyannote, NeMo, cloud APIs, anything that emits standard output
- **Local-first** — one SQLite file, no server, no accounts; audio never
  leaves your machine
- **Honest about uncertainty** — three confidence bands (matched / review /
  new) with a built-in review-and-confirm workflow, instead of pretending
  the model is always right
- **Calibrate on *your* audio** — `speakerdex calibrate` measures similarity
  distributions on voices you know and recommends thresholds, rather than
  hardcoding magic numbers

## Why this doesn't already exist

Cross-file speaker identity is a years-old open ask: pyannote's maintainer
says it works ["not out of the box"](https://github.com/pyannote/pyannote-audio/discussions/1085),
and AssemblyAI's official answer is
["build it yourself"](https://www.assemblyai.com/docs/faq/do-you-offer-cross-file-speaker-identification)
with a third-party embedding model and a vector database. The commercial
version of this primitive is [priced per voiceprint](https://www.pyannote.ai/pricing),
closed, and cloud-only. Everyone who needs it hand-rolls the same clustering
script. This is that script, done properly, once.

## Measured, not promised

Numbers from the two evaluation runs in `docs/`. These are small corpora —
treat them as evidence, not benchmarks. Results on other material are the most
useful contribution you can make.

| | LibriVox (3 readers, 9 files) | Podcast, conversational (5 voices, 3 eps) |
|---|---|---|
| Genuine same-voice similarity | 0.73–0.93 | 0.909–0.916 |
| Worst impostor similarity | 0.25 | 0.288 |
| Identification | 9/9 leave-one-out | 8/8 clusters, incl. a diarizer label flip |
| Default thresholds held | yes (calibrate recommended 0.55/0.37) | yes, with wide margin |

Full write-ups: [docs/first-real-run.md](docs/first-real-run.md) and
[docs/real-world-testing.md](docs/real-world-testing.md). Both record their own
caveats and failure modes.

## How it works

```
 audio + diarization (RTTM / WhisperX JSON)
        │
        ▼
 ┌─────────────────┐   per-cluster speaker embeddings (ECAPA-TDNN)
 │  embed_clusters │──────────────────────────────┐
 └─────────────────┘                              ▼
                                        ┌──────────────────┐
 ┌─────────────────┐  identity centroids│     matcher      │
 │  voice registry │───────────────────▶│ greedy 1:1, with │
 │    (SQLite)     │◀───────────────────│ confidence bands │
 └─────────────────┘  new voiceprints,  └──────────────────┘
                      assignments                 │
                                                  ▼
                          matched / review / new  per cluster
```

Matches land in one of three bands: **matched** (confident), **review**
(probable — confirm with one command), or **new** (auto-enroll as `Unknown-N`
if you ask). Embedding backends are pluggable; the registry refuses to mix
embeddings from different models.

## Quickstart

```bash
pip install "speakerdex[ecapa]"    # the ECAPA extra pulls in torch + speechbrain
```

The core install has no ML dependencies; the `[ecapa]` extra adds the embedding
model you need to work with real audio.

```bash
# 1. Create a registry next to your project
speakerdex init
```

```bash
# 2. See who is in a diarized file — which SPEAKER_NN is which person?
speakerdex clusters ep01.json
```

```
ep01.json: 3 clusters, 152 segments, 618.4s labelled speech

SPEAKER_02    361.0s  58.4%   85 seg
    [11:41]  would be like a hand that the model could use in the…
    [06:01]  and then anthropic followed up with a bunch of cool stuff after…

SPEAKER_01    150.7s  24.4%   37 seg
    [04:26]  Maybe you can or cannot speak to just how you personally feel…
    [04:51]  That's kind of what I want to cover is that bigger landscape…

Pass --audio to also see the likely registry match for each cluster.
Enroll with:  speakerdex enroll "<name>" <audio> --diarization ep01.json --cluster <LABEL>
```

Identify people from the **transcript previews**, not the speaking time: on
interview formats the guest routinely outspeaks the host. (RTTM carries no
transcript, so previews only appear for WhisperX JSON.)

```bash
# 3. Enroll a voice — from a solo clip…
speakerdex enroll "Alex" alex_intro.wav

#    …or straight out of a diarized episode
speakerdex enroll "Adam Stacoviak" ep01.mp3 --diarization ep01.json --cluster SPEAKER_01
#   Enrolled 'Adam Stacoviak' from ep01.mp3 (65.2s of speech, 1 voiceprint(s) total)
```

Once the registry has someone in it, `clusters --audio` dry-runs the match
before you commit to anything — it writes nothing:

```bash
speakerdex clusters ep02.json --audio ep02.mp3
```

```
SPEAKER_02    360.6s  58.0%   96 seg   no match (closest: Adam Stacoviak, sim=0.134)
SPEAKER_01    147.8s  23.8%   36 seg   likely: Adam Stacoviak (sim=0.916, matched)
SPEAKER_00    113.1s  18.2%   28 seg   no match (closest: Adam Stacoviak, sim=0.149)
```

```bash
# 4. Process files: clusters resolve to stable names
speakerdex process ep02.mp3 --diarization ep02.json --enroll-unknowns

#    …or a whole season at once: ep01.mp3 pairs with ep01.json / ep01.rttm
speakerdex process-dir episodes/ --enroll-unknowns
```

```bash
# 5. Curate the registry as you learn who people are
speakerdex ls
#      1  Adam Stacoviak  (1 voiceprints, seen in 3 file(s))
#      2  Unknown-1  (1 voiceprints, seen in 2 file(s))

speakerdex rename Unknown-1 "Jordan"
speakerdex review                                  # borderline matches, if any
speakerdex confirm ep02.mp3 SPEAKER_01 "Jordan"
```

```bash
# 6. Write identity names back into a WhisperX transcript
speakerdex process ep02.mp3 -d ep02.json -o ep02_named.json
```

Works with any diarizer that emits RTTM, and with WhisperX JSON directly
(word-level speakers are relabelled too; the original cluster label is
preserved as `speaker_cluster`).

`process-dir` walks files in sorted filename order, which matters with
`--enroll-unknowns`: the first file containing a new voice is the one that
creates its `Unknown-N` identity, and later files match against it. Files
already recorded in the registry are skipped, so re-running as a season grows
only costs the new episodes — pass `--force` to reprocess everything.

## Library use

```python
from speakerdex import Registry, process_file, enroll_from_audio
from speakerdex.adapters import load_segments
from speakerdex.embeddings import get_backend

registry = Registry("speakerdex.db")
backend = get_backend("ecapa")

enroll_from_audio("Alex", "alex.wav", registry, backend)
decisions = process_file("ep02.wav", load_segments("ep02.rttm"), registry, backend)
```

## What speakerdex is not

It is not a diarizer (it consumes diarization output), not a transcription
tool, and not a cloud service. It does one thing: persistent speaker identity
across files. If you need "who said what" within a single file, WhisperX and
pyannote already do that well.

## Design decisions

- **Bring-your-own-diarization.** Diarization quality is its own arms race; speakerdex consumes its output rather than competing with it. Anything that emits RTTM or WhisperX JSON works.
- **Local-first.** SQLite file, no server, no accounts, audio never leaves your machine. Linear scan over identity centroids is plenty fast up to thousands of identities; an ANN index is a later optimization, not a v1 requirement.
- **Human-in-the-loop by design.** Similarity thresholds are honest about uncertainty: the review band plus `confirm`/`rename`/`merge` makes curation a first-class workflow instead of pretending the model is always right.
- **Backend-locked registries.** Embeddings from different models live in incompatible spaces; a registry records its backend and refuses to mix.

## Threshold guidance

Cosine thresholds are backend- and corpus-dependent. The defaults
(`match ≥ 0.55`, `review ≥ 0.40` for ECAPA) are a conservative starting point:
studio podcasts can run higher; noisy field recordings lower. Tune with
`--match-threshold` / `--review-threshold`, and measure rather than guess with
`speakerdex calibrate` — see [docs/real-world-testing.md](docs/real-world-testing.md).

## Status & roadmap

**v0.1.0.** It works and it is tested — 60 model-free tests plus two documented
real-world runs — but it is young, and the APIs may still move.

- [ ] Threshold calibration from confirmed assignments
- [ ] `speakerdex stats` — registry health (per-identity voiceprint spread, drift)
- [ ] Additional backends (wespeaker, pyannote/embedding) behind extras
- [x] Batch mode: `speakerdex process-dir` over a season of episodes
- [ ] Export/import registries (share a cast registry for a show)
- [ ] Representative sample per identity — store one `(source, start, end)` with each voiceprint so `ls` can show what `Unknown-3` actually sounds like instead of sending you back to the transcript; needs a schema migration, so deferred rather than bolted on
- [ ] Time-range exclusion (e.g. `--skip 0:00-2:10`) — baked-in sponsor reads are recurring voices and become permanent identities; whether the rule belongs per-file, per-run or stored in the registry is an open design question
- [ ] Margin reporting — speakerdex knows how close every call came to the thresholds and should say so ("nearest miss: 0.29 below match"); folds into `speakerdex stats` above rather than being its own command
- [ ] Overlap-aware embedding exclusion (skip segments where diarization reports overlapping speech)
- [ ] Benchmarks on public multi-episode corpora (e.g. VoxConverse, This American Life dataset)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The single most valuable contribution is
a `speakerdex calibrate` report from a corpus that isn't in `docs/` yet — that
is what turns the threshold defaults from two data points into guidance.

## Development

```bash
uv sync --extra dev          # core + test tooling (no ML deps needed for tests)
uv run pytest                # model-free test suite (synthetic voices)
uv run ruff check .
```

The test suite uses a spectral fake backend on synthesized harmonic "voices," so it runs in seconds with zero ML dependencies — the ECAPA backend is only needed for real audio.

## License

Apache-2.0
