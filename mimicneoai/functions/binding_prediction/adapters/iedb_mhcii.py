"""IEDB MHC-II NNalign predictor adapter."""

from __future__ import annotations

from pathlib import Path

from mimicneoai.functions.binding_prediction.schema import BindingPrediction, PredictionJob

from .base import (
    PredictorAdapter,
    ensure_job_dirs,
    reusable_normalized_output,
    read_tabular_after_header,
    run_logged_command,
    write_error_predictions,
    write_ok_predictions,
    write_peptide_fasta,
)


class IedbMhciiAdapter(PredictorAdapter):
    """Run IEDB MHC-II NNalign on fixed-length peptides."""

    supported_algorithms = frozenset({"NNalign"})
    default_chunk_size = 8000

    def run_job(self, job: PredictionJob) -> Path:
        if reusable_normalized_output(job, self.config.resume):
            return job.normalized_path
        ensure_job_dirs(job)
        try:
            write_peptide_fasta(job.input_path, job.peptides)
            cmd = [
                self.config.iedb_mhcii_python_bin,
                self.config.iedb_mhcii_script,
                "nn_align",
                format_iedb_mhcii_allele(job.hla_allele),
                str(job.input_path),
                str(job.peptide_length),
            ]
            run_logged_command(
                cmd,
                Path(self.config.iedb_mhcii_cwd),
                job.log_path,
                stdout_path=job.raw_path,
                config=self.config,
            )
            rows = read_tabular_after_header(job.raw_path, "allele\t")
            predictions = []
            for row in rows:
                peptide = (row.get("peptide") or "").strip()
                if not peptide:
                    continue
                predictions.append(
                    BindingPrediction(
                        peptide=peptide,
                        hla_allele=job.hla_allele,
                        algorithm=job.algorithm,
                        mhc_class=job.mhc_class,
                        peptide_length=job.peptide_length,
                        ic50=row.get("ic50", ""),
                        ic50_type="IEDB IC50(nM)",
                        percentile=row.get("percentile_rank", ""),
                        percentile_type="IEDB percentile_rank",
                        raw_rank=row.get("adjusted_rank", ""),
                        raw_rank_type="IEDB adjusted_rank",
                        raw_file=str(job.raw_path),
                    )
                )
            return write_ok_predictions(job, predictions)
        except Exception as exc:  # noqa: BLE001
            return write_error_predictions(job, exc)


def format_iedb_mhcii_allele(allele: str) -> str:
    """Convert internal HLA-II allele notation to IEDB's expected notation."""

    allele = allele.strip()
    if allele.startswith("HLA-"):
        formatted = allele
    else:
        formatted = "HLA-" + allele
    if any(gene in formatted for gene in ("DPA1", "DQA1")):
        formatted = formatted.replace("-DPB1", "/DPB1").replace("-DQB1", "/DQB1")
    return formatted
