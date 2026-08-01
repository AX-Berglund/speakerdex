"""Batch processing: resolve a whole folder of episodes in one pass.

Given a directory laid out as::

    season1/
      ep01.wav  ep01.rttm
      ep02.wav  ep02.json
      ep03.wav                 <- no diarization partner: reported, not fatal

each audio file is paired with the diarization file sharing its stem and run
through :func:`speakerdex.pipeline.process_file`, against one registry.

Ordering
--------
Files are processed in sorted filename order, and that order is part of the
contract rather than an implementation detail. With ``enroll_unknowns``, the
first file containing a previously unseen voice is the file that *creates* the
``Unknown-N`` identity; every later file matches against it instead. So the
order decides which file an identity is born in, which number it gets, and
which recording its first voiceprint comes from — and since that first
voiceprint seeds the centroid later matches are scored against, a noisy first
appearance propagates. Sorted order makes this deterministic and reproducible:
the same folder always yields the same registry. Rename episodes if you want a
cleaner recording to be the one an identity is anchored to, or enroll the
important voices by hand first.

Idempotency
-----------
A file whose (source, cluster) assignments are already in the registry is
skipped, so re-running over a growing folder only costs the new episodes.
``force=True`` reprocesses everything; assignments are upserted, so nothing is
duplicated. Skipping is keyed on the source path exactly as recorded, so the
same file reached by a different path spelling counts as unprocessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .adapters import load_segments
from .calibrate import AUDIO_EXTS  # one source of truth for "what counts as audio"
from .embeddings import EmbeddingBackend
from .pipeline import ProcessConfig, process_file
from .registry import Assignment, Registry
from .types import MATCHED, NEW, REVIEW, MatchDecision

# Checked in order, so .rttm wins when a file has both partners.
DIARIZATION_EXTS = (".rttm", ".json")

PROCESSED = "processed"
ALREADY_PROCESSED = "already-processed"
NO_DIARIZATION = "no-diarization"

_NOTES = {
    ALREADY_PROCESSED: "already processed",
    NO_DIARIZATION: "no diarization file",
}


def is_visible(path: Path) -> bool:
    """"_"/"." prefixed entries are scratch (caches, AppleDouble files), not input."""
    return not path.name.startswith(("_", "."))


def find_diarization(audio: Path) -> Path | None:
    """The diarization file sharing this audio file's stem, if one exists."""
    for ext in DIARIZATION_EXTS:
        candidate = audio.with_suffix(ext)
        if candidate.is_file():
            return candidate
    return None


@dataclass
class FilePair:
    audio: Path
    diarization: Path | None


def pair_files(directory: str | Path) -> list[FilePair]:
    """Pair every audio file in ``directory`` with its diarization, in sorted order.

    Non-recursive: subdirectories are ignored entirely in v1.
    """
    pairs = []
    for entry in sorted(Path(directory).iterdir()):
        if entry.is_dir() or not is_visible(entry):
            continue
        if entry.suffix.lower() not in AUDIO_EXTS:
            continue
        pairs.append(FilePair(entry, find_diarization(entry)))
    return pairs


@dataclass
class FileResult:
    audio: Path
    diarization: Path | None
    status: str  # PROCESSED | ALREADY_PROCESSED | NO_DIARIZATION
    decisions: list[MatchDecision] = field(default_factory=list)

    @property
    def from_registry(self) -> bool:
        """True when decisions were replayed from stored assignments, not recomputed."""
        return self.status == ALREADY_PROCESSED

    @property
    def note(self) -> str | None:
        return _NOTES.get(self.status)


def _replay(assignment: Assignment) -> MatchDecision:
    """Present a stored assignment as a decision, so skipped files still report."""
    return MatchDecision(
        cluster=assignment.cluster,
        identity_id=assignment.identity_id,
        identity_name=assignment.identity_name,
        similarity=assignment.similarity,
        status=assignment.status,
    )


@dataclass
class BatchReport:
    directory: Path
    results: list[FileResult]

    @property
    def processed(self) -> list[FileResult]:
        return [r for r in self.results if r.status == PROCESSED]

    @property
    def skipped(self) -> list[FileResult]:
        return [r for r in self.results if r.status != PROCESSED]

    def cluster_counts(self) -> dict[str, int]:
        """Clusters by status across every file in the report, replayed ones included."""
        counts = {MATCHED: 0, REVIEW: 0, NEW: 0}
        for result in self.results:
            for decision in result.decisions:
                if decision.status in counts:
                    counts[decision.status] += 1
        return counts

    def roster(self) -> dict[str, list[str]]:
        """identity name -> the files it appeared in, in processing order."""
        seen: dict[str, list[str]] = {}
        for result in self.results:
            for decision in result.decisions:
                if decision.identity_name is None:
                    continue
                files = seen.setdefault(decision.identity_name, [])
                if result.audio.name not in files:
                    files.append(result.audio.name)
        return {name: seen[name] for name in sorted(seen)}

    def summary(self) -> str:
        lines: list[str] = []
        for result in self.results:
            note = f"  [{result.note}]" if result.note else ""
            lines.append(f"{result.audio.name}{note}")
            for decision in sorted(result.decisions, key=lambda d: d.cluster):
                lines.append(f"  {decision}")

        if not self.results:
            return f"No audio files found in {self.directory}"

        breakdown = ", ".join(
            f"{n} {_NOTES[status]}"
            for status in (ALREADY_PROCESSED, NO_DIARIZATION)
            if (n := sum(1 for r in self.skipped if r.status == status))
        )
        counts = self.cluster_counts()
        lines += [
            "",
            f"Files: {len(self.processed)} processed, {len(self.skipped)} skipped"
            + (f" ({breakdown})" if breakdown else ""),
            f"Clusters: {counts[MATCHED]} matched, {counts[REVIEW]} review, {counts[NEW]} new",
        ]

        roster = self.roster()
        lines.append("Identities seen:" if roster else "Identities seen: none")
        width = max((len(name) for name in roster), default=0)
        for name, files in roster.items():
            lines.append(f"  {name:<{width}}  {', '.join(files)}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "directory": str(self.directory),
            "files": [
                {
                    "audio": str(r.audio),
                    "diarization": str(r.diarization) if r.diarization else None,
                    "status": r.status,
                    "decisions": [
                        {
                            "cluster": d.cluster,
                            "identity": d.identity_name,
                            "similarity": round(d.similarity, 4),
                            "status": d.status,
                            # not stored on assignments, so unknown when replayed
                            "speech_seconds": None if r.from_registry else round(d.total_speech, 1),
                        }
                        for d in sorted(r.decisions, key=lambda d: d.cluster)
                    ],
                }
                for r in self.results
            ],
            "summary": {
                "processed": len(self.processed),
                "skipped": len(self.skipped),
                "already_processed": sum(
                    1 for r in self.skipped if r.status == ALREADY_PROCESSED
                ),
                "no_diarization": sum(1 for r in self.skipped if r.status == NO_DIARIZATION),
                "clusters": self.cluster_counts(),
            },
            "roster": self.roster(),
        }


def process_dir(
    directory: str | Path,
    registry: Registry,
    backend: EmbeddingBackend,
    config: ProcessConfig | None = None,
    force: bool = False,
) -> BatchReport:
    """Process every paired audio file in ``directory`` against one registry.

    Files are handled in sorted filename order (see the module docstring: with
    ``enroll_unknowns`` that order determines which file an identity is born
    in). Audio with no diarization partner is reported as skipped rather than
    raising, so one stray file cannot abort a season.
    """
    config = config or ProcessConfig()
    registry.check_backend(backend.name)  # fail before doing any work, not halfway through

    results: list[FileResult] = []
    for pair in pair_files(directory):
        if pair.diarization is None:
            results.append(FileResult(pair.audio, None, NO_DIARIZATION))
            continue

        existing = registry.assignments_for(str(pair.audio))
        if existing and not force:
            replayed = [_replay(a) for a in existing]
            results.append(
                FileResult(pair.audio, pair.diarization, ALREADY_PROCESSED, replayed)
            )
            continue

        segments = load_segments(pair.diarization)
        decisions = process_file(pair.audio, segments, registry, backend, config)
        results.append(FileResult(pair.audio, pair.diarization, PROCESSED, decisions))

    return BatchReport(Path(directory), results)
