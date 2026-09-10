"""IEDB MHC-I predictor adapter: SMM, SMMPMBEC, PickPocket."""

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


METHOD_BY_ALGORITHM = {
    "SMM": "smm",
    "SMMPMBEC": "smmpmbec",
    "PickPocket": "pickpocket",
}


class IedbMhciAdapter(PredictorAdapter):
    """Run IEDB MHC-I command-line predictors on fixed-length peptides."""

    supported_algorithms = frozenset(METHOD_BY_ALGORITHM)
    default_chunk_size = 1000

    def run_job(self, job: PredictionJob) -> Path:
        if reusable_normalized_output(job, self.config.resume):
            return job.normalized_path
        ensure_job_dirs(job)
        try:
            write_peptide_fasta(job.input_path, job.peptides)
            cmd = [
                self.config.python_bin,
                self.config.iedb_mhci_script,
                METHOD_BY_ALGORITHM[job.algorithm],
                job.hla_allele,
                str(job.peptide_length),
                str(job.input_path),
            ]
            run_logged_command(
                cmd,
                Path(self.config.iedb_mhci_cwd),
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
                        percentile=row.get("rank", ""),
                        percentile_type="IEDB rank",
                        raw_rank=row.get("rank", ""),
                        raw_rank_type="IEDB rank",
                        raw_file=str(job.raw_path),
                    )
                )
            return write_ok_predictions(job, predictions)
        except Exception as exc:  # noqa: BLE001 - keep per-chunk failure recoverable
            return write_error_predictions(job, exc)
