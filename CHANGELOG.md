# Changelog

All notable changes to this project are documented here. This project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 — 2026-08-02

First public release.

### Pipeline

- Per-cluster speaker embeddings from audio + diarization, duration-weighted
  and L2-normalized, with speech capped per cluster so cost stays bounded.
- Greedy one-to-one matching of clusters to identities within a file: two
  clusters cannot both claim the same person.
- Three confidence bands — `matched`, `review`, `new` — with configurable
  `--match-threshold` / `--review-threshold`.
- Optional `--enroll-unknowns` (auto-create `Unknown-N`) and `--reinforce`
  (add high-confidence matches as fresh voiceprints).
- Pluggable embedding backends; ECAPA-TDNN via SpeechBrain behind the
  `[ecapa]` extra.

### Registry

- SQLite store of identities, voiceprints and assignments; no server, no
  accounts.
- Backend lock-in: a registry records the embedding backend that created it
  and refuses to mix incompatible embedding spaces.
- Curation primitives: `rename`, `merge`, and a `review`/`confirm` queue for
  borderline matches.

### CLI

- `init`, `enroll`, `clusters`, `process`, `process-dir`, `calibrate`, `ls`,
  `rename`, `merge`, `review`, `confirm`, `version`.
- `clusters` lists every cluster in a diarization file with speaking time,
  share and — for WhisperX JSON — transcript previews, so `enroll --cluster`
  no longer requires guessing a label. With `--audio` it dry-runs the match
  against the registry, writing nothing.
- Enrolling a cluster label that does not exist fails with the labels that do,
  seconds included, so the command is fixable from the error alone.
- `enroll` reports the seconds of speech captured and warns below 10s, where
  voiceprints stop being reliable.
- `--json` output on `process`, `process-dir`, `clusters` and `calibrate`.

### Calibration

- `calibrate` measures same- vs different-speaker cosine distributions over a
  labelled `voices/<name>/*.wav` tree, prints an ASCII histogram and
  leave-one-out identification accuracy, and recommends thresholds measured on
  your own audio.
- `scripts/setup_voices.py` builds a starter calibration set from public-domain
  LibriVox recordings.

### Batch

- `process-dir` resolves a whole folder in one command, pairing `ep01.mp3` with
  `ep01.rttm` or `ep01.json` by stem.
- Deterministic sorted-filename order, which decides where an identity is born
  under `--enroll-unknowns`.
- Idempotent: already-processed files are skipped unless `--force`, so
  re-running a growing season only costs the new episodes.
- Run summary with per-band cluster counts and a roster of which identity
  appeared in which files.

### Adapters

- RTTM and WhisperX JSON in; RTTM and WhisperX JSON out, with identity names
  written back into transcripts (word-level speakers included, original cluster
  label preserved as `speaker_cluster`).

### Docs

- Two documented real-world evaluations: a LibriVox calibration run and a
  three-episode conversational podcast run, both with measured similarity
  distributions, thresholds and known failure modes.
- `docs/real-world-testing.md` walks through calibrating and testing on your
  own material.
