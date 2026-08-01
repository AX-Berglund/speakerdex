# Contributing to speakerdex

## The most valuable contribution: calibration reports

speakerdex's default thresholds (`match ≥ 0.55`, `review ≥ 0.40`) currently rest
on **two corpora** — a LibriVox read-speech set and one three-episode podcast,
both written up in `docs/`. That is enough to show the approach works and
nowhere near enough to call it guidance.

If you run speakerdex on material of your own, the output of:

```bash
speakerdex calibrate voices/
```

is the single most useful thing you can send. Open an issue with:

- the calibrate output block (histogram, distributions, recommended thresholds)
- roughly what the audio is: number of speakers, recording conditions, language,
  read vs conversational, remote vs studio
- whether the defaults held, and where they broke if they didn't

No audio needed — the numbers are the contribution. Cases where speakerdex does
*badly* are more valuable than cases where it does well; the first real run
already showed that studio-consistent audio flatters the scores, and the honest
picture needs the other kind.

## Development setup

```bash
uv sync --extra dev          # core + test tooling, no ML dependencies
uv run pytest
uv run ruff check .
```

Working on the ECAPA backend or real audio additionally needs `--extra ecapa`,
which pulls in torch and speechbrain.

## Tests are model-free, on purpose

The suite runs in about a second with **zero ML dependencies**. Each synthetic
"voice" is a harmonic tone at a distinct fundamental frequency, and
`fake-spectral` (in `tests/conftest.py`) embeds audio by its spectral band
profile — so the same voice yields similar embeddings across files and different
voices don't. That is exactly the property speakerdex depends on, without
downloading a model in CI.

Please keep new tests on this backend. If you are testing something that only
manifests with real embeddings, say so in the test docstring and gate it.

One gotcha worth knowing: `fake-spectral` scores two *different* synthetic
voices at roughly 0.67 and the same voice at ~1.0 — a different scale from
ECAPA, whose defaults the CLI ships. Tests that put a single voice in a file
alone must therefore pass thresholds calibrated for that backend (see
`tests/test_batch.py`), or a lone stranger clears 0.55 and false-matches.

## Style

- `ruff check .` must pass; line length is 100.
- Match the surrounding code: comments explain *why*, not *what*.
- New CLI logic belongs in its own module (`batch.py`, `clusters.py`,
  `calibrate.py`), with `cli.py` doing only argument wiring and printing.

## Pull requests

CI runs ruff and the test suite on Python 3.10 and 3.12. Please make sure both
pass locally first. For anything that changes behaviour, a test that fails
before your change is worth more than a paragraph of description.
