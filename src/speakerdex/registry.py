"""The voice registry: SQLite-backed store of identities, voiceprints and assignments.

Schema
------
identities   one row per known person (stable across files)
voiceprints  L2-normalized embeddings attached to an identity
assignments  the resolved mapping (source file, cluster) -> identity, with confidence
meta         key/value store (schema version, embedding backend/dim lock-in)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .embeddings import l2_normalize

SCHEMA_VERSION = "1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS identities (
  id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  notes TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS voiceprints (
  id INTEGER PRIMARY KEY,
  identity_id INTEGER NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
  embedding BLOB NOT NULL,
  dim INTEGER NOT NULL,
  duration REAL NOT NULL DEFAULT 0,
  source TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS assignments (
  id INTEGER PRIMARY KEY,
  source TEXT NOT NULL,
  cluster TEXT NOT NULL,
  identity_id INTEGER REFERENCES identities(id) ON DELETE SET NULL,
  similarity REAL NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(source, cluster)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Identity:
    id: int
    name: str
    notes: str
    voiceprint_count: int = 0


@dataclass
class Assignment:
    """A recorded (source file, cluster) -> identity resolution."""

    source: str
    cluster: str
    identity_id: int | None
    identity_name: str | None
    similarity: float
    status: str


class Registry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(_SCHEMA)
        self.conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)",
            (SCHEMA_VERSION,),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> Registry:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- meta ---------------------------------------------------------------

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self.conn.commit()

    def check_backend(self, backend_name: str) -> None:
        """Voiceprints from different embedding models don't mix; lock the backend."""
        existing = self.get_meta("backend")
        if existing is None:
            self.set_meta("backend", backend_name)
        elif existing != backend_name:
            raise ValueError(
                f"registry was created with backend {existing!r}; refusing to mix in "
                f"embeddings from {backend_name!r} (embedding spaces are incompatible)"
            )

    # -- identities ---------------------------------------------------------

    def enroll(self, name: str, notes: str = "") -> int:
        cur = self.conn.execute(
            "INSERT INTO identities(name, notes, created_at) VALUES(?, ?, ?)",
            (name, notes, _now()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def identity_by_name(self, name: str) -> Identity | None:
        row = self.conn.execute(
            "SELECT id, name, notes FROM identities WHERE name = ?", (name,)
        ).fetchone()
        return Identity(*row) if row else None

    def identities(self) -> list[Identity]:
        rows = self.conn.execute(
            "SELECT i.id, i.name, i.notes, COUNT(v.id) "
            "FROM identities i LEFT JOIN voiceprints v ON v.identity_id = i.id "
            "GROUP BY i.id ORDER BY i.name"
        ).fetchall()
        return [Identity(*row) for row in rows]

    def file_counts(self) -> dict[int, int]:
        """identity id -> how many distinct source files it has been assigned in."""
        rows = self.conn.execute(
            "SELECT identity_id, COUNT(DISTINCT source) FROM assignments "
            "WHERE identity_id IS NOT NULL GROUP BY identity_id"
        ).fetchall()
        return dict(rows)

    def rename(self, old: str, new: str) -> None:
        cur = self.conn.execute("UPDATE identities SET name = ? WHERE name = ?", (new, old))
        if cur.rowcount == 0:
            raise KeyError(f"no identity named {old!r}")
        self.conn.commit()

    def merge(self, source_name: str, dest_name: str) -> None:
        """Move all voiceprints/assignments from source into dest, delete source."""
        src = self.identity_by_name(source_name)
        dst = self.identity_by_name(dest_name)
        if src is None or dst is None:
            raise KeyError("both identities must exist to merge")
        self.conn.execute(
            "UPDATE voiceprints SET identity_id = ? WHERE identity_id = ?", (dst.id, src.id)
        )
        self.conn.execute(
            "UPDATE assignments SET identity_id = ? WHERE identity_id = ?", (dst.id, src.id)
        )
        self.conn.execute("DELETE FROM identities WHERE id = ?", (src.id,))
        self.conn.commit()

    # -- voiceprints --------------------------------------------------------

    def add_voiceprint(
        self, identity_id: int, embedding: np.ndarray, duration: float = 0.0, source: str = ""
    ) -> int:
        vec = l2_normalize(embedding)
        cur = self.conn.execute(
            "INSERT INTO voiceprints(identity_id, embedding, dim, duration, source, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (identity_id, vec.tobytes(), vec.size, duration, source, _now()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def centroids(self) -> dict[int, np.ndarray]:
        """Per-identity mean voiceprint (L2-normalized), keyed by identity id."""
        rows = self.conn.execute("SELECT identity_id, embedding, dim FROM voiceprints").fetchall()
        grouped: dict[int, list[np.ndarray]] = {}
        for identity_id, blob, dim in rows:
            vec = np.frombuffer(blob, dtype=np.float32, count=dim)
            grouped.setdefault(identity_id, []).append(vec)
        return {
            identity_id: l2_normalize(np.stack(vecs).mean(axis=0))
            for identity_id, vecs in grouped.items()
        }

    # -- assignments --------------------------------------------------------

    def record_assignment(
        self, source: str, cluster: str, identity_id: int | None, similarity: float, status: str
    ) -> None:
        self.conn.execute(
            "INSERT INTO assignments(source, cluster, identity_id, similarity, status, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(source, cluster) DO UPDATE SET "
            "identity_id = excluded.identity_id, similarity = excluded.similarity, "
            "status = excluded.status, created_at = excluded.created_at",
            (source, cluster, identity_id, similarity, status, _now()),
        )
        self.conn.commit()

    def assignments_for(self, source: str) -> list[Assignment]:
        """Every assignment already recorded for one source file (empty if unprocessed)."""
        rows = self.conn.execute(
            "SELECT a.source, a.cluster, a.identity_id, i.name, a.similarity, a.status "
            "FROM assignments a LEFT JOIN identities i ON i.id = a.identity_id "
            "WHERE a.source = ? ORDER BY a.cluster",
            (source,),
        ).fetchall()
        return [Assignment(*row) for row in rows]

    def pending_reviews(self) -> list[tuple[str, str, str | None, float]]:
        """(source, cluster, identity_name, similarity) for review-band assignments."""
        return self.conn.execute(
            "SELECT a.source, a.cluster, i.name, a.similarity "
            "FROM assignments a LEFT JOIN identities i ON i.id = a.identity_id "
            "WHERE a.status = 'review' ORDER BY a.source, a.cluster"
        ).fetchall()

    def confirm(self, source: str, cluster: str, name: str) -> None:
        """Resolve a review item: bind (source, cluster) to the named identity."""
        ident = self.identity_by_name(name)
        if ident is None:
            raise KeyError(f"no identity named {name!r}")
        cur = self.conn.execute(
            "UPDATE assignments SET identity_id = ?, status = 'matched' "
            "WHERE source = ? AND cluster = ?",
            (ident.id, source, cluster),
        )
        if cur.rowcount == 0:
            raise KeyError(f"no assignment for ({source!r}, {cluster!r})")
        self.conn.commit()
