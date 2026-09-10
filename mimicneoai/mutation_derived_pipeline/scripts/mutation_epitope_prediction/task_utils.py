"""Prediction task construction and de-duplication utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BindingTask:
    """One unique peptide-HLA-algorithm prediction task."""

    peptide: str
    hla_allele: str
    algorithm: str
    mhc_class: str


def build_binding_tasks(
    peptides: list[str],
    hla_alleles: list[str],
    algorithms: list[str],
    mhc_class: str,
) -> list[BindingTask]:
    """Build de-duplicated binding prediction tasks."""

    tasks = {
        BindingTask(
            peptide=peptide,
            hla_allele=hla_allele,
            algorithm=algorithm,
            mhc_class=mhc_class,
        )
        for peptide in peptides
        for hla_allele in hla_alleles
        for algorithm in algorithms
        if peptide and hla_allele and algorithm
    }
    return sorted(tasks, key=lambda x: (x.mhc_class, x.algorithm, x.hla_allele, x.peptide))


def split_tasks(tasks: list[BindingTask], chunk_size: int) -> list[list[BindingTask]]:
    """Split prediction tasks for batch execution."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return [tasks[i : i + chunk_size] for i in range(0, len(tasks), chunk_size)]
