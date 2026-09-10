"""HLA result parsing helpers for mutation-derived epitope prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from itertools import product
import re


HLA_I_PRESENTING_GENES = ("A", "B", "C", "E", "F", "G")
HLA_DRB_PRESENTING_GENES = ("DRB1", "DRB3", "DRB4", "DRB5")
HLA_DQ_ALPHA_GENES = ("DQA1",)
HLA_DQ_BETA_GENES = ("DQB1",)
HLA_DP_ALPHA_GENES = ("DPA1",)
HLA_DP_BETA_GENES = ("DPB1",)


@dataclass(frozen=True)
class HlaAlleles:
    """Patient HLA alleles split by MHC class."""

    mhc_i: tuple[str, ...]
    mhc_ii: tuple[str, ...]


def parse_hlahd_result(hla_result_file: Path) -> HlaAlleles:
    """Parse an HLA-HD ``*_final.result.txt`` file."""

    hla_result_file = Path(hla_result_file)
    gene_to_alleles: dict[str, list[str]] = {}
    with hla_result_file.open() as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if not fields:
                continue
            gene = fields[0]
            alleles: list[str] = []
            for raw_allele in fields[1:]:
                if not raw_allele or raw_allele in {"-", "Not typed"}:
                    continue
                allele = normalize_hla_allele(raw_allele)
                if allele:
                    alleles.append(allele)
            if alleles:
                gene_to_alleles[gene] = list(dict.fromkeys(alleles))

    mhc_i: list[str] = []
    for gene in HLA_I_PRESENTING_GENES:
        mhc_i.extend(gene_to_alleles.get(gene, []))

    # HLA-II presentation requires the correct molecule model:
    # - DR specificity is represented by DRB alleles. DRA is near invariant and
    #   should not be emitted as a standalone prediction target.
    # - DQ and DP require alpha-beta pairing.
    # - Non-presenting/ancillary class-II genes such as DMA/DMB/DOA/DOB are not
    #   included in epitope binding tasks.
    mhc_ii: list[str] = []
    for gene in HLA_DRB_PRESENTING_GENES:
        mhc_ii.extend(_strip_hla_prefix(a) for a in gene_to_alleles.get(gene, []))

    dpa = _collect_alleles(gene_to_alleles, HLA_DP_ALPHA_GENES)
    dpb = _collect_alleles(gene_to_alleles, HLA_DP_BETA_GENES)
    dqa = _collect_alleles(gene_to_alleles, HLA_DQ_ALPHA_GENES)
    dqb = _collect_alleles(gene_to_alleles, HLA_DQ_BETA_GENES)
    mhc_ii.extend(f"{a}-{b}" for a, b in product(dpa, dpb))
    mhc_ii.extend(f"{a}-{b}" for a, b in product(dqa, dqb))

    return HlaAlleles(
        mhc_i=tuple(dict.fromkeys(mhc_i)),
        mhc_ii=tuple(dict.fromkeys(mhc_ii)),
    )


def normalize_hla_allele(allele: str) -> str:
    """Normalize one HLA allele to the resolution used by predictors."""

    allele = allele.strip()
    if not allele:
        return allele
    if not allele.startswith("HLA-"):
        allele = f"HLA-{allele}"
    gene, _, fields = allele.partition("*")
    parts = fields.split(":")
    if len(parts) >= 2 and re.fullmatch(r"\d+", parts[0]) and re.fullmatch(r"\d+", parts[1]):
        return f"{gene}*{parts[0]}:{parts[1]}"
    return ""


def _strip_hla_prefix(allele: str) -> str:
    """Strip the optional ``HLA-`` prefix from class-II allele labels."""

    return allele[4:] if allele.startswith("HLA-") else allele


def _collect_alleles(gene_to_alleles: dict[str, list[str]], genes: tuple[str, ...]) -> list[str]:
    """Collect class-II alleles from selected genes with ``HLA-`` stripped."""

    alleles: list[str] = []
    for gene in genes:
        alleles.extend(_strip_hla_prefix(a) for a in gene_to_alleles.get(gene, []))
    return alleles
