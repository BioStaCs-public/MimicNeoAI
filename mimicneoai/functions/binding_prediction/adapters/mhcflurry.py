"""MHCflurry adapter for class I affinity and presentation prediction."""

from __future__ import annotations

import csv
from pathlib import Path

from mimicneoai.functions.binding_prediction.schema import BindingPrediction, PredictionJob

from .base import (
    PredictorAdapter,
    ensure_job_dirs,
    reusable_normalized_output,
    run_logged_command,
    write_error_predictions,
    write_ok_predictions,
)


class MhcflurryAdapter(PredictorAdapter):
    """Run MHCflurry on fixed peptide-HLA pairs."""

    supported_algorithms = frozenset({"MHCflurry", "MHCflurryEL"})
    default_chunk_size = 50000

    def run_job(self, job: PredictionJob) -> Path:
        if reusable_normalized_output(job, self.config.resume):
            return job.normalized_path
        ensure_job_dirs(job)
        try:
            if job.mhc_class != "MHC-I":
                raise ValueError(f"{job.algorithm} only supports MHC-I tasks, got {job.mhc_class}")
            write_mhcflurry_input(job.input_path, job.hla_allele, job.peptides)
            cmd = [
                self.config.mhcflurry_predict_bin,
                str(job.input_path),
                "--out",
                str(job.raw_path),
                "--no-throw",
            ]
            if job.algorithm == "MHCflurry":
                cmd.append("--affinity-only")
            else:
                cmd.append("--no-flanking")
            run_logged_command(cmd, None, job.log_path, config=self.config)
            rows = list(csv.DictReader(job.raw_path.open()))
            predictions = []
            output_algorithms = job.output_algorithms or (job.algorithm,)
            for row in rows:
                peptide = (row.get("peptide") or "").strip()
                if not peptide:
                    continue
                for algorithm in output_algorithms:
                    if algorithm == "MHCflurryEL":
                        score = row.get("mhcflurry_presentation_score", "")
                        score_type = "MHCflurry presentation score"
                        percentile = row.get("mhcflurry_presentation_percentile", "")
                        percentile_type = "MHCflurry presentation percentile"
                    else:
                        score = ""
                        score_type = ""
                        percentile = row.get("mhcflurry_affinity_percentile", "")
                        percentile_type = "MHCflurry affinity percentile"
                    predictions.append(
                        BindingPrediction(
                            peptide=peptide,
                            hla_allele=job.hla_allele,
                            algorithm=algorithm,
                            mhc_class=job.mhc_class,
                            peptide_length=job.peptide_length,
                            ic50=row.get("mhcflurry_affinity", ""),
                            ic50_type="MHCflurry affinity(nM)",
                            score=score,
                            score_type=score_type,
                            percentile=percentile,
                            percentile_type=percentile_type,
                            raw_rank=percentile,
                            raw_rank_type=percentile_type,
                            raw_file=str(job.raw_path),
                        )
                    )
            return write_ok_predictions(job, predictions)
        except Exception as exc:  # noqa: BLE001
            return write_error_predictions(job, exc)


def write_mhcflurry_input(path: Path, hla_allele: str, peptides: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["allele", "peptide"])
        writer.writeheader()
        for peptide in peptides:
            writer.writerow({"allele": hla_allele, "peptide": peptide})
