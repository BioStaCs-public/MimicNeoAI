"""NetMHCIIpan 4.3 adapter for BA and EL outputs."""

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


class NetMHCIIpanAdapter(PredictorAdapter):
    """Run NetMHCIIpan once per HLA-II allele/pair and peptide length."""

    supported_algorithms = frozenset({"NetMHCIIpan", "NetMHCIIpanEL"})
    default_chunk_size = 30000

    def run_job(self, job: PredictionJob):
        if reusable_normalized_output(job, self.config.resume):
            return job.normalized_path
        ensure_job_dirs(job)
        try:
            write_peptide_lines(job.input_path, job.peptides)
            relative_input_path = job.input_path.relative_to(job.key_dir)
            output_algorithms = job.output_algorithms or (job.algorithm,)
            el_only = self.config.netmhc_el_only and set(output_algorithms) == {"NetMHCIIpanEL"}
            cmd = [
                self.config.netmhciipan_bin,
                "-inptype",
                "1",
                "-f",
                str(relative_input_path),
                "-a",
                format_netmhciipan_allele(job.hla_allele),
            ]
            if not el_only:
                cmd.append("-BA")
            run_logged_command(cmd, job.key_dir, job.log_path, stdout_path=job.raw_path, config=self.config)
            rows = parse_netmhc_stdout_table(job.raw_path)
            if not rows:
                raise ValueError(f"NetMHCIIpan returned no parseable rows for {job.hla_allele}")
            predictions = []
            for row in rows:
                peptide = (row.get("Peptide") or "").strip()
                if not peptide:
                    continue
                for algorithm in output_algorithms:
                    if algorithm == "NetMHCIIpanEL":
                        score = row.get("Score", "") or row.get("Score_EL", "")
                        percentile = row.get("Rank", "") or row.get("%Rank_EL", "")
                        raw_rank = row.get("Rank", "") or row.get("%Rank_EL", "")
                        score_type = "NetMHCIIpan EL Score"
                        percentile_type = "NetMHCIIpan EL %Rank"
                        raw_rank_type = "NetMHCIIpan EL %Rank"
                    else:
                        score = row.get("Score_BA", "")
                        percentile = row.get("Rank_BA", "") or row.get("%Rank_BA", "")
                        raw_rank = row.get("Rank_BA", "") or row.get("%Rank_BA", "")
                        score_type = "NetMHCIIpan BA Score"
                        percentile_type = "NetMHCIIpan BA %Rank"
                        raw_rank_type = "NetMHCIIpan BA %Rank"
                    predictions.append(
                        BindingPrediction(
                            peptide=peptide,
                            hla_allele=job.hla_allele,
                            algorithm=algorithm,
                            mhc_class=job.mhc_class,
                            peptide_length=job.peptide_length,
                            ic50=row.get("nM", "") or row.get("Affinity(nM)", ""),
                            ic50_type="NetMHCIIpan Affinity(nM)",
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


def format_netmhciipan_allele(allele: str) -> str:
    """Convert internal HLA-II notation to NetMHCIIpan notation.

    NetMHCIIpan accepts DR alleles as ``DRB1_0101``. DP/DQ alpha-beta pairs
    use compact HLA-prefixed pseudo-sequence names such as
    ``HLA-DPA10201-DPB10501`` and ``HLA-DQA10505-DQB10301``.
    """

    allele = allele.strip()
    if "-" in allele and any(gene in allele for gene in ("DPA1", "DQA1")):
        parts = []
        for part in allele.split("-"):
            part = part.strip()
            if part.startswith("HLA-"):
                part = part[4:]
            parts.append(part.replace("*", "").replace(":", ""))
        return "HLA-" + "-".join(parts)

    parts = []
    for part in allele.split("/"):
        part = part.strip()
        if part.startswith("HLA-"):
            part = part[4:]
        part = part.replace("*", "_").replace(":", "")
        parts.append(part)
    return "-".join(parts)
