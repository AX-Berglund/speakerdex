"""speakerdex command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from . import __version__
from .adapters import load_segments, write_whisperx
from .matcher import MatchConfig
from .pipeline import ProcessConfig, enroll_from_audio, process_file
from .registry import Registry

app = typer.Typer(
    name="speakerdex",
    help="Persistent speaker identity across files: a local voice registry.",
    no_args_is_help=True,
)

DEFAULT_DB = "speakerdex.db"

_db_option = typer.Option(DEFAULT_DB, "--db", help="Path to the registry database.")


def _backend(name: str):
    from .embeddings import get_backend

    return get_backend(name)


@app.callback()
def _main() -> None:
    """speakerdex: the same voice gets the same name, across every file."""


@app.command()
def version() -> None:
    """Print the speakerdex version."""
    typer.echo(__version__)


@app.command()
def init(db: str = _db_option) -> None:
    """Create a new (empty) registry database."""
    path = Path(db)
    existed = path.exists()
    Registry(path).close()
    typer.echo(f"{'Opened existing' if existed else 'Created'} registry at {path}")


@app.command()
def enroll(
    name: str = typer.Argument(..., help="Identity name, e.g. 'Alex'."),
    audio: Path = typer.Argument(..., exists=True, help="Audio file with this person's speech."),
    diarization: Path | None = typer.Option(
        None, "--diarization", "-d", help="Diarization file (RTTM or WhisperX JSON)."
    ),
    cluster: str | None = typer.Option(
        None, "--cluster", "-c", help="Cluster label to enroll from (with --diarization)."
    ),
    fmt: str | None = typer.Option(None, "--format", help="Diarization format: rttm | whisperx."),
    backend: str = typer.Option("ecapa", "--backend", help="Embedding backend."),
    db: str = _db_option,
) -> None:
    """Enroll a voice. Without --diarization, the whole file is assumed to be one speaker."""
    segments = load_segments(diarization, fmt) if diarization else None
    with Registry(db) as registry:
        enroll_from_audio(name, audio, registry, _backend(backend), segments, cluster)
        count = next(i.voiceprint_count for i in registry.identities() if i.name == name)
    typer.echo(f"Enrolled {name!r} from {audio.name} ({count} voiceprint(s) total)")


@app.command()
def process(
    audio: Path = typer.Argument(..., exists=True, help="Audio file to process."),
    diarization: Path = typer.Option(
        ..., "--diarization", "-d", help="Diarization file (RTTM or WhisperX JSON)."
    ),
    fmt: str | None = typer.Option(None, "--format", help="Diarization format: rttm | whisperx."),
    backend: str = typer.Option("ecapa", "--backend", help="Embedding backend."),
    match_threshold: float = typer.Option(0.55, help="Cosine similarity for a confident match."),
    review_threshold: float = typer.Option(0.40, help="Below this: treated as a new speaker."),
    enroll_unknowns: bool = typer.Option(
        False, "--enroll-unknowns", help="Auto-enroll unmatched clusters as Unknown-N."
    ),
    reinforce: bool = typer.Option(
        False, "--reinforce", help="Add high-confidence matches as fresh voiceprints."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write relabelled WhisperX JSON here (whisperx input only)."
    ),
    json_out: bool = typer.Option(False, "--json", help="Print decisions as JSON."),
    db: str = _db_option,
) -> None:
    """Match every diarized speaker cluster in a file against the registry."""
    segments = load_segments(diarization, fmt)
    config = ProcessConfig(
        match=MatchConfig(match_threshold=match_threshold, review_threshold=review_threshold),
        enroll_unknowns=enroll_unknowns,
        reinforce=reinforce,
    )
    with Registry(db) as registry:
        decisions = process_file(audio, segments, registry, _backend(backend), config)

    if json_out:
        typer.echo(
            json.dumps(
                [
                    {
                        "cluster": d.cluster,
                        "identity": d.identity_name,
                        "similarity": round(d.similarity, 4),
                        "status": d.status,
                        "speech_seconds": round(d.total_speech, 1),
                    }
                    for d in decisions
                ],
                indent=2,
            )
        )
    else:
        for d in sorted(decisions, key=lambda d: d.cluster):
            typer.echo(str(d))

    if output is not None:
        mapping = {d.cluster: d.identity_name for d in decisions if d.identity_name}
        write_whisperx(diarization, output, mapping)
        typer.echo(f"Relabelled transcript written to {output}")


@app.command("process-dir")
def process_dir(
    directory: Path = typer.Argument(
        ..., exists=True, file_okay=False, help="Folder of audio + diarization files."
    ),
    backend: str = typer.Option("ecapa", "--backend", help="Embedding backend."),
    match_threshold: float = typer.Option(0.55, help="Cosine similarity for a confident match."),
    review_threshold: float = typer.Option(0.40, help="Below this: treated as a new speaker."),
    enroll_unknowns: bool = typer.Option(
        False, "--enroll-unknowns", help="Auto-enroll unmatched clusters as Unknown-N."
    ),
    reinforce: bool = typer.Option(
        False, "--reinforce", help="Add high-confidence matches as fresh voiceprints."
    ),
    force: bool = typer.Option(
        False, "--force", help="Reprocess files that already have assignments."
    ),
    json_out: bool = typer.Option(False, "--json", help="Print the whole run as one JSON object."),
    db: str = _db_option,
) -> None:
    """Process a folder of files in sorted order: ep01.wav pairs with ep01.rttm/.json.

    Already-processed files are skipped unless --force; audio with no
    diarization partner is reported as skipped, not an error.
    """
    from .batch import process_dir as run_batch

    config = ProcessConfig(
        match=MatchConfig(match_threshold=match_threshold, review_threshold=review_threshold),
        enroll_unknowns=enroll_unknowns,
        reinforce=reinforce,
    )
    with Registry(db) as registry:
        report = run_batch(directory, registry, _backend(backend), config, force=force)

    typer.echo(json.dumps(report.to_dict(), indent=2) if json_out else report.summary())


@app.command()
def calibrate(
    voices_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False,
        help="Directory of known voices: voices/<speaker_name>/*.wav (2+ files per speaker).",
    ),
    backend: str = typer.Option("ecapa", "--backend", help="Embedding backend."),
    json_out: bool = typer.Option(False, "--json", help="Print the report as JSON."),
) -> None:
    """Measure similarity distributions on YOUR audio and recommend thresholds."""
    from .calibrate import calibrate as run_calibration

    report = run_calibration(voices_dir, _backend(backend))
    if json_out:
        typer.echo(
            json.dumps(
                {
                    "n_speakers": report.n_speakers,
                    "n_files": report.n_files,
                    "match_threshold": report.match_threshold,
                    "review_threshold": report.review_threshold,
                    "top1_accuracy": round(report.top1_accuracy, 4),
                    "same_median": round(float(np.median(report.same)), 4),
                    "diff_median": round(float(np.median(report.diff)), 4),
                    "warnings": report.warnings,
                },
                indent=2,
            )
        )
    else:
        typer.echo(report.summary())


@app.command("ls")
def list_identities(db: str = _db_option) -> None:
    """List enrolled identities."""
    with Registry(db) as registry:
        identities = registry.identities()
    if not identities:
        typer.echo("No identities enrolled yet. Try: speakerdex enroll <name> <audio>")
        raise typer.Exit()
    for ident in identities:
        typer.echo(f"{ident.id:4d}  {ident.name}  ({ident.voiceprint_count} voiceprints)")


@app.command()
def rename(old: str, new: str, db: str = _db_option) -> None:
    """Rename an identity (e.g. Unknown-3 -> 'Sam')."""
    with Registry(db) as registry:
        registry.rename(old, new)
    typer.echo(f"Renamed {old!r} -> {new!r}")


@app.command()
def merge(source: str, dest: str, db: str = _db_option) -> None:
    """Merge one identity into another (voiceprints and assignments move to dest)."""
    with Registry(db) as registry:
        registry.merge(source, dest)
    typer.echo(f"Merged {source!r} into {dest!r}")


@app.command()
def review(db: str = _db_option) -> None:
    """List assignments in the review band (probable but unconfirmed matches)."""
    with Registry(db) as registry:
        rows = registry.pending_reviews()
    if not rows:
        typer.echo("Nothing pending review.")
        raise typer.Exit()
    for source, cluster, name, sim in rows:
        typer.echo(f"{source}  {cluster}  ->  {name or '?'}  (sim={sim:.3f})")
    typer.echo("\nConfirm with: speakerdex confirm <source> <cluster> <name>")


@app.command()
def confirm(source: str, cluster: str, name: str, db: str = _db_option) -> None:
    """Resolve a review item by binding (source, cluster) to an identity."""
    with Registry(db) as registry:
        registry.confirm(source, cluster, name)
    typer.echo(f"Confirmed: {cluster} in {source} is {name!r}")


if __name__ == "__main__":
    app()
