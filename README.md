# speakerdex

**Persistent speaker identity across files.** A local voice registry that sits on top of any diarization pipeline and makes sure the same voice gets the same name — across episodes, recordings and projects.

> Status: **pre-alpha, private**. APIs will change without notice.

## The problem

Every diarization tool (pyannote, WhisperX, NeMo, cloud APIs) labels speakers `SPEAKER_00`, `SPEAKER_01` — and forgets them the moment the file ends. Process episode 2 and the same host is suddenly `SPEAKER_03`. There is no maintained open-source tool that answers: *"is this the same person I heard yesterday?"*

- pyannote's maintainer: cross-file identity works ["not out of the box"](https://github.com/pyannote/pyannote-audio/discussions/1085)
- AssemblyAI's official answer: [DIY it yourself](https://www.assemblyai.com/docs/faq/do-you-offer-cross-file-speaker-identification) with a third-party model and a vector DB
- The commercial version of this primitive is [priced per voiceprint](https://www.pyannote.ai/pricing), closed, and cloud-only

speakerdex is the missing piece: **bring your own diarization, keep your audio local, get stable identities.**

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

Matches land in one of three bands: **matched** (confident), **review** (probable — confirm with one command), or **new** (auto-enroll as `Unknown-N` if you ask). Embedding backends are pluggable; the registry refuses to mix embeddings from different models.

## Quickstart

```bash
pip install "speakerdex[ecapa]"

# 1. Create a registry next to your project
speakerdex init

# 2. Enroll a voice — from a solo clip…
speakerdex enroll "Alex" alex_intro.wav

#    …or straight out of a diarized episode
speakerdex enroll "Sam" ep01.wav --diarization ep01.rttm --cluster SPEAKER_02

# 3. Process new files: clusters resolve to stable names
speakerdex process ep02.wav --diarization ep02.rttm --enroll-unknowns
#   SPEAKER_00 -> Alex (matched, sim=0.71)
#   SPEAKER_01 -> Sam (matched, sim=0.64)
#   SPEAKER_02 -> Unknown-1 (new, sim=0.22)

#    …or do a whole season at once: ep01.wav pairs with ep01.rttm / ep01.json
speakerdex process-dir season1/ --enroll-unknowns
#   ep01.wav
#     SPEAKER_00 -> Alex (matched, sim=0.71)
#     SPEAKER_01 -> Unknown-1 (new, sim=0.22)
#   ep02.wav
#     SPEAKER_00 -> Unknown-1 (matched, sim=0.68)
#   ep03.wav  [no diarization file]
#
#   Files: 2 processed, 1 skipped (1 no diarization file)
#   Clusters: 2 matched, 0 review, 1 new
#   Identities seen:
#     Alex       ep01.wav
#     Unknown-1  ep01.wav, ep02.wav

# 4. Curate the registry as you learn who people are
speakerdex ls
speakerdex rename Unknown-1 "Jordan"
speakerdex review          # see borderline matches
speakerdex confirm ep02.wav SPEAKER_01 "Sam"

# 5. Write identity names back into a WhisperX transcript
speakerdex process ep02.wav -d ep02.json -o ep02_named.json
```

Works with any diarizer that emits RTTM, and with WhisperX JSON directly (word-level speakers are relabelled too; the original cluster label is preserved as `speaker_cluster`).

`process-dir` walks files in sorted filename order, which matters with `--enroll-unknowns`: the first file containing a new voice is the one that creates its `Unknown-N` identity, and later files match against it. Files already recorded in the registry are skipped, so re-running as a season grows only costs the new episodes — pass `--force` to reprocess everything.

## Library use

```python
from speakerdex import Registry, process_file, enroll_from_audio
from speakerdex.embeddings import get_backend
from speakerdex.adapters import load_segments

registry = Registry("speakerdex.db")
backend = get_backend("ecapa")

enroll_from_audio("Alex", "alex.wav", registry, backend)
decisions = process_file("ep02.wav", load_segments("ep02.rttm"), registry, backend)
```

## Design decisions

- **Bring-your-own-diarization.** Diarization quality is its own arms race; speakerdex consumes its output rather than competing with it. Anything that emits RTTM or WhisperX JSON works.
- **Local-first.** SQLite file, no server, no accounts, audio never leaves your machine. Linear scan over identity centroids is plenty fast up to thousands of identities; an ANN index is a later optimization, not a v1 requirement.
- **Human-in-the-loop by design.** Similarity thresholds are honest about uncertainty: the review band plus `confirm`/`rename`/`merge` makes curation a first-class workflow instead of pretending the model is always right.
- **Backend-locked registries.** Embeddings from different models live in incompatible spaces; a registry records its backend and refuses to mix.

## Threshold guidance

Cosine thresholds are backend- and corpus-dependent. The defaults (`match ≥ 0.55`, `review ≥ 0.40` for ECAPA) are a conservative starting point: studio podcasts can run higher; noisy field recordings lower. Tune with `--match-threshold` / `--review-threshold`. Automatic calibration from confirmed assignments is on the roadmap.

## Roadmap

- [ ] Threshold calibration from confirmed assignments
- [ ] `speakerdex stats` — registry health (per-identity voiceprint spread, drift)
- [ ] Additional backends (wespeaker, pyannote/embedding) behind extras
- [x] Batch mode: `speakerdex process-dir` over a season of episodes
- [ ] Export/import registries (share a cast registry for a show)
- [ ] Overlap-aware embedding exclusion (skip segments where diarization reports overlapping speech)
- [ ] Benchmarks on public multi-episode corpora (e.g. VoxConverse, This American Life dataset)

## Development

```bash
uv sync --extra dev          # core + test tooling (no ML deps needed for tests)
uv run pytest                # model-free test suite (synthetic voices)
uv run ruff check .
```

The test suite uses a spectral fake backend on synthesized harmonic "voices," so it runs in seconds with zero ML dependencies — the ECAPA backend is only needed for real audio.

## License

Apache-2.0
