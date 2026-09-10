"""NetMHCpan 4.2 adapter for BA and EL outputs."""

from __future__ import annotations

from mimicneoai.functions.binding_prediction.schema import BindingPrediction, PredictionJob

from .base import (
    PredictorAdapter,
    ensure_job_dirs,
    reusable_normalized_output,
    parse_netmhc_stdout_table,
    run_logged_command,
    write_error_predictions,
    write_ok_predictions,
    write_peptide_lines,
)


class NetMHCpanAdapter(PredictorAdapter):
    """Run NetMHCpan once per HLA and peptide length."""

    supported_algorithms = frozenset({"NetMHCpan", "NetMHCpanEL"})
    default_chunk_size = 50000

    def run_job(self, job: PredictionJob):
        if reusable_normalized_output(job, self.config.resume):
            return job.normalized_path
        ensure_job_dirs(job)
        try:
            write_peptide_lines(job.input_path, job.peptides)
            relative_input_path = job.input_path.relative_to(job.key_dir)
            output_algorithms = job.output_algorithms or (job.algorithm,)
            el_only = self.config.netmhc_el_only and set(output_algorithms) == {"NetMHCpanEL"}
            cmd = [
                self.config.netmhcpan_bin,
                "-p",
                "-f",
                str(relative_input_path),
                "-a",
                format_netmhcpan_allele(job.hla_allele),
            ]
            if not el_only:
                cmd.append("-BA")
            run_logged_command(cmd, job.key_dir, job.log_path, stdout_path=job.raw_path, config=self.config)
            rows = parse_netmhc_stdout_table(job.raw_path)
            if not rows:
                raise ValueError(f"NetMHCpan returned no parseable rows for {job.hla_allele}")
            predictions = []
            for row in rows:
                peptide = (row.get("Peptide") or "").strip()
                if not peptide:
                    continue
                for algorithm in output_algorithms:
                    if algorithm == "NetMHCpanEL":
                        score = row.get("EL-score", "") or row.get("Score", "")
                        percentile = row.get("EL_Rank", "") or row.get("Rank", "") or row.get("%Rank", "")
                        raw_rank = row.get("EL_Rank", "") or row.get("Rank", "") or row.get("%Rank", "")
                        score_type = "NetMHCpan EL Score"
                        percentile_type = "NetMHCpan EL %Rank"
                        raw_rank_type = "NetMHCpan EL %Rank"
                    else:
                        score = row.get("BA-score", "") or row.get("BA_score", "") or row.get("Score_BA", "")
                        percentile = row.get("BA_Rank", "") or row.get("%Rank_BA", "")
                        raw_rank = row.get("BA_Rank", "") or row.get("%Rank_BA", "")
                        score_type = "NetMHCpan BA Score"
                        percentile_type = "NetMHCpan BA %Rank"
                        raw_rank_type = "NetMHCpan BA %Rank"
                    predictions.append(
                        BindingPrediction(
                            peptide=peptide,
                            hla_allele=job.hla_allele,
                            algorithm=algorithm,
                            mhc_class=job.mhc_class,
                            peptide_length=job.peptide_length,
                            ic50=row.get("Aff(nM)", "") or row.get("nM", ""),
                            ic50_type="NetMHCpan Aff(nM)",
                            score=score,
                            score_type=score_type,
                            percentile=percentile,
                            percentile_type=percentile_type,
                            raw_rank=raw_rank,
                            raw_rank_type=raw_rank_type,
                            raw_file=str(job.raw_path),
                        )
                    )
            return write_ok_predictions(job, predictions)
        except Exception as exc:  # noqa: BLE001
            return write_error_predictions(job, exc)


def format_netmhcpan_allele(allele: str) -> str:
    """Convert HLA-A*02:01 to NetMHCpan's common HLA-A02:01 form."""

    allele = allele.strip()
    if allele.startswith("HLA-"):
        prefix, value = allele.split("*", 1)
        return prefix + value
    return allele.replace("*", "")
