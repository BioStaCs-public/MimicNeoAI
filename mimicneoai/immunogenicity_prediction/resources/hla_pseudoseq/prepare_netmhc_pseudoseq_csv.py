#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


RESOURCE_ROOT = Path(__file__).resolve().parent
DEFAULT_CLASS1_RAW = RESOURCE_ROOT / "local/raw/netmhcpan/MHC_pseudo.dat"
DEFAULT_CLASS2_RAW = RESOURCE_ROOT / "local/raw/netmhciipan/pseudosequence.2023.dat"
DEFAULT_OUT_DIR = RESOURCE_ROOT / "local"


def normalize_class1(raw: str) -> str:
    allele = raw.strip().replace(" ", "")
    match = re.fullmatch(r"HLA-([ABC])(\d{2}):(\d{2})", allele)
    if match:
        return f"HLA-{match.group(1)}*{match.group(2)}:{match.group(3)}"
    return allele


def normalize_class2(raw: str) -> str:
    allele = raw.strip().replace(" ", "")
    match = re.fullmatch(r"(DRB[1-5])_(\d{2})(\d{2})", allele)
    if match:
        return f"HLA-{match.group(1)}*{match.group(2)}:{match.group(3)}"

    match = re.fullmatch(
        r"HLA-(DPA1|DPB1|DQA1|DQB1)(\d{2})(\d{2})-"
        r"(DPA1|DPB1|DQA1|DQB1)(\d{2})(\d{2})",
        allele,
    )
    if match:
        return (
            f"HLA-{match.group(1)}*{match.group(2)}:{match.group(3)}-"
            f"{match.group(4)}*{match.group(5)}:{match.group(6)}"
        )
    return allele


def write_pseudoseq_csv(
    raw_path: Path,
    out_path: Path,
    *,
    mhc_class: str,
    normalizer,
) -> int:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw_allele, pseudo = line.split()[:2]
            allele = normalizer(raw_allele)
            if allele in seen:
                continue
            seen.add(allele)
            rows.append(
                {
                    "allele": allele,
                    "pseudo_sequence": pseudo,
                    "mhc_class": mhc_class,
                    "raw_allele": raw_allele,
                    "source_file": str(raw_path),
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["allele", "pseudo_sequence", "mhc_class", "raw_allele", "source_file"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--class1-raw", default=DEFAULT_CLASS1_RAW, type=Path)
    parser.add_argument("--class2-raw", default=DEFAULT_CLASS2_RAW, type=Path)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    args = parser.parse_args()

    class1_out = args.out_dir / "netmhcpan_class1_allele_to_pseudoseq.csv"
    class2_out = args.out_dir / "netmhciipan_class2_allele_to_pseudoseq.csv"

    class1_count = write_pseudoseq_csv(
        args.class1_raw,
        class1_out,
        mhc_class="I",
        normalizer=normalize_class1,
    )
    class2_count = write_pseudoseq_csv(
        args.class2_raw,
        class2_out,
        mhc_class="II",
        normalizer=normalize_class2,
    )
    print(
        {
            "class1_out": str(class1_out),
            "class1_rows": class1_count,
            "class2_out": str(class2_out),
            "class2_rows": class2_count,
        }
    )


if __name__ == "__main__":
    main()
