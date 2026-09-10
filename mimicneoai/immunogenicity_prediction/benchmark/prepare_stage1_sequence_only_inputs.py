#!/usr/bin/env python3
"""Build leakage-controlled pooled Stage 1 inputs from source-specific TSVs."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path


def parse_source_spec(value: str) -> tuple[str, Path, Path]:
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected SOURCE=TRAIN_TSV,VALIDATION_TSV")
    paths = parts[1].split(",", 1)
    if len(paths) != 2:
        raise argparse.ArgumentTypeError("Expected SOURCE=TRAIN_TSV,VALIDATION_TSV")
    return parts[0], Path(paths[0]), Path(paths[1])


def require_table(path: Path) -> list[str]:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle, delimiter="\t"))
    required = {"peptide", "hla", "label"}
    missing = required - set(header)
    if missing:
        raise KeyError(f"{path} missing columns: {sorted(missing)}")
    return header


def phla_key(row: dict[str, str]) -> str:
    return f"{row['peptide'].strip().upper()}|{row['hla'].strip()}"


def update_label_index(
    connection: sqlite3.Connection,
    split: str,
    path: Path,
    excluded_peptides: set[str] | None = None,
) -> tuple[int, int]:
    rows = 0
    excluded = 0
    batch: list[tuple[str, str, int]] = []
    sql = (
        "INSERT INTO labels(split, phla, label_mask) VALUES (?, ?, ?) "
        "ON CONFLICT(split, phla) DO UPDATE SET label_mask = label_mask | excluded.label_mask"
    )
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            peptide = row["peptide"].strip().upper()
            if excluded_peptides is not None and peptide in excluded_peptides:
                excluded += 1
                continue
            label = int(row["label"])
            if label not in {0, 1}:
                raise ValueError(f"Unsupported label={label} in {path}")
            batch.append((split, phla_key(row), 1 << label))
            rows += 1
            if len(batch) == 100_000:
                connection.executemany(sql, batch)
                connection.commit()
                batch.clear()
    if batch:
        connection.executemany(sql, batch)
        connection.commit()
    return rows, excluded


def validation_peptides(specs: list[tuple[str, Path, Path]]) -> set[str]:
    peptides: set[str] = set()
    for _, _, validation_path in specs:
        with validation_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                peptides.add(row["peptide"].strip().upper())
    return peptides


def conflict_keys(connection: sqlite3.Connection, split: str) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT phla FROM labels WHERE split = ? AND label_mask = 3",
            (split,),
        )
    }


def write_pooled_split(
    connection: sqlite3.Connection,
    specs: list[tuple[str, Path, Path]],
    split: str,
    output: Path,
    fields: list[str],
    excluded_peptides: set[str] | None,
    conflicts: set[str],
) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    connection.execute("DELETE FROM written")
    connection.commit()
    counts: Counter[tuple[str, int]] = Counter()
    removed_quarantine = 0
    removed_conflict = 0
    removed_duplicate = 0
    with output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for source, train_path, validation_path in specs:
            path = train_path if split == "train" else validation_path
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    peptide = row["peptide"].strip().upper()
                    if excluded_peptides is not None and peptide in excluded_peptides:
                        removed_quarantine += 1
                        continue
                    key = phla_key(row)
                    if key in conflicts:
                        removed_conflict += 1
                        continue
                    inserted = connection.execute(
                        "INSERT OR IGNORE INTO written(phla) VALUES (?)",
                        (key,),
                    ).rowcount
                    if not inserted:
                        removed_duplicate += 1
                        continue
                    row["peptide"] = peptide
                    row["hla"] = row["hla"].strip()
                    row["antigen_class"] = source
                    row["pooled_source"] = source
                    writer.writerow(row)
                    counts[(source, int(row["label"]))] += 1
            connection.commit()
    return {
        "split": split,
        "output": str(output),
        "rows": sum(counts.values()),
        "removed_validation_peptide_rows": removed_quarantine,
        "removed_label_conflict_rows": removed_conflict,
        "removed_duplicate_phla_rows": removed_duplicate,
        "source_label_counts": {
            f"{source}|{label}": count
            for (source, label), count in sorted(counts.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        type=parse_source_spec,
        help="Repeat SOURCE=TRAIN_TSV,VALIDATION_TSV for every antigen class.",
    )
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args()

    specs = args.source
    headers: list[str] = []
    for _, train_path, validation_path in specs:
        for path in (train_path, validation_path):
            for field in require_table(path):
                if field not in headers:
                    headers.append(field)
    for field in ("antigen_class", "pooled_source"):
        if field not in headers:
            headers.append(field)

    args.outdir.mkdir(parents=True, exist_ok=True)
    heldout_peptides = validation_peptides(specs)
    with tempfile.TemporaryDirectory(prefix="mimicneoai_stage1_pool_") as temp_dir:
        connection = sqlite3.connect(Path(temp_dir) / "index.sqlite")
        connection.execute(
            "CREATE TABLE labels(split TEXT, phla TEXT, label_mask INTEGER, PRIMARY KEY(split, phla))"
        )
        connection.execute("CREATE TABLE written(phla TEXT PRIMARY KEY)")
        scan_rows = {}
        for source, train_path, validation_path in specs:
            validation_rows, _ = update_label_index(connection, "validation", validation_path)
            train_rows, excluded = update_label_index(
                connection,
                "train",
                train_path,
                excluded_peptides=heldout_peptides,
            )
            scan_rows[source] = {
                "train_rows_after_validation_peptide_quarantine": train_rows,
                "train_rows_removed_by_validation_peptide_quarantine": excluded,
                "validation_rows": validation_rows,
            }

        train_conflicts = conflict_keys(connection, "train")
        validation_conflicts = conflict_keys(connection, "validation")
        train_output = args.outdir / "pooled_stage1_mixed_decoy_v3.train.model_input.tsv"
        validation_output = args.outdir / "pooled_stage1_mixed_decoy_v3.validation.model_input.tsv"
        outputs = [
            write_pooled_split(
                connection,
                specs,
                "train",
                train_output,
                headers,
                heldout_peptides,
                train_conflicts,
            ),
            write_pooled_split(
                connection,
                specs,
                "validation",
                validation_output,
                headers,
                None,
                validation_conflicts,
            ),
        ]
        connection.close()

    manifest = {
        "method": "global validation-peptide quarantine, conflicting-label pHLA removal, deterministic same-label pHLA deduplication",
        "source_order": [source for source, _, _ in specs],
        "validation_unique_peptides": len(heldout_peptides),
        "train_label_conflict_unique_phla": len(train_conflicts),
        "validation_label_conflict_unique_phla": len(validation_conflicts),
        "scan_rows": scan_rows,
        "outputs": outputs,
    }
    (args.outdir / "pooled_stage1_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
