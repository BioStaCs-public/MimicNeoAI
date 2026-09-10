"""Small, dependency-free I/O helpers used by the mimicry package."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Iterator


def open_text(path: Path, mode: str = "rt"):
    """Open plain or gzip-compressed text based on the filename suffix."""

    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, object]:
    """Return a reproducible file identity suitable for manifests."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(resolved),
    }


def read_tsv(path: Path) -> Iterator[tuple[int, dict[str, str]]]:
    """Yield one-based data-row numbers and TSV records."""

    with open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"TSV has no header: {path}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"TSV contains duplicate column names: {path}")
        for row_number, row in enumerate(reader, start=2):
            yield row_number, {str(key): str(value or "") for key, value in row.items()}


def read_tsv_header(path: Path) -> tuple[str, ...]:
    """Return a validated TSV header without reading the data rows."""

    with open_text(path) as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            fields = tuple(next(reader))
        except StopIteration as exc:
            raise ValueError(f"TSV has no header: {path}") from exc
    if not fields or not any(fields):
        raise ValueError(f"TSV has no header: {path}")
    if len(fields) != len(set(fields)):
        raise ValueError(f"TSV contains duplicate column names: {path}")
    return fields


def require_tsv_columns(path: Path, required: Iterable[str]) -> tuple[str, ...]:
    """Fail closed when a versioned table does not satisfy its schema."""

    fields = read_tsv_header(path)
    missing = sorted(set(required) - set(fields))
    if missing:
        raise ValueError(f"TSV is missing required columns {missing}: {path}")
    return fields


def write_tsv(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, object]],
) -> int:
    """Atomically write a plain or gzip-compressed TSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".tmp.gz" if path.suffix == ".gz" else ".tmp"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=suffix, dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary_name)
    count = 0
    try:
        with open_text(temporary_path, "wt") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                delimiter="\t",
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
                count += 1
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return count


def write_json(path: Path, data: dict[str, object]) -> None:
    """Atomically write a deterministic JSON document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(
            json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def read_json(path: Path) -> dict[str, object]:
    """Read a JSON object and reject non-object roots."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value
