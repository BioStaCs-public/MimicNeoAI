"""MHCnuggets adapter for class I and class II affinity prediction."""

from __future__ import annotations

import csv
import pickle
import re
from bisect import bisect_left, bisect_right
from pathlib import Path

from mimicneoai.functions.binding_prediction.schema import BindingPrediction, PredictionJob

from .base import (
    PredictorAdapter,
    ensure_job_dirs,
    reusable_normalized_output,
    run_logged_command,
    write_error_predictions,
    write_ok_predictions,
    write_peptide_lines,
)


CLASS_BY_ALGORITHM = {
    "MHCnuggetsI": "I",
    "MHCnuggetsII": "II",
}


class MhcnuggetsAdapter(PredictorAdapter):
    """Run MHCnuggets with the same model-selection behavior as pVACtools."""

    supported_algorithms = frozenset(CLASS_BY_ALGORITHM)
    default_chunk_size = 3000
    class_i_min_length = 8
    class_i_max_length = 15

    def run_job(self, job: PredictionJob) -> Path:
        if reusable_normalized_output(job, self.config.resume):
            return job.normalized_path
        ensure_job_dirs(job)
        try:
            expected_class = "MHC-I" if job.algorithm == "MHCnuggetsI" else "MHC-II"
            if job.mhc_class != expected_class:
                raise ValueError(f"{job.algorithm} expects {expected_class}, got {job.mhc_class}")
            if job.algorithm == "MHCnuggetsI" and not (
                self.class_i_min_length <= job.peptide_length <= self.class_i_max_length
            ):
                raise ValueError(
                    f"MHCnuggetsI supports class-I peptide length "
                    f"{self.class_i_min_length}-{self.class_i_max_length}, got {job.peptide_length}"
                )
            write_peptide_lines(job.input_path, job.peptides)
            rank_error = ""
            run_logged_command(
                build_mhcnuggets_command(job, rank_output=False, config=self.config),
                Path(self.config.mhcnuggets_cwd),
                job.log_path,
                config=self.config,
            )
            rows = list(csv.DictReader(job.raw_path.open()))
            rank_by_peptide = {}
            if self.config.mhcnuggets_rank_output:
                try:
                    predictor_allele = read_closest_mhcnuggets_allele(job.log_path)
                    rank_by_peptide = calculate_mhcnuggets_ranks(
                        rows,
                        job.mhc_class,
                        predictor_allele,
                        Path(self.config.mhcnuggets_cwd),
                    )
                except Exception as exc:  # noqa: BLE001 - retain IC50 when rank resources fail
                    rank_error = str(exc)
            rows_by_peptide: dict[str, dict[str, str]] = {}
            for row in rows:
                peptide = normalize_peptide(row.get("peptide", ""))
                if peptide and peptide not in rows_by_peptide:
                    rows_by_peptide[peptide] = row

            predictions = []
            missing_peptides = []
            for peptide in job.peptides:
                normalized_peptide = normalize_peptide(peptide)
                row = rows_by_peptide.get(normalized_peptide)
                if row is None:
                    missing_peptides.append(peptide)
                    continue
                rank = rank_by_peptide.get(normalized_peptide, "")
                percentile = human_proteome_percentile(rank)
                predictions.append(
                    BindingPrediction(
                        peptide=peptide,
                        hla_allele=job.hla_allele,
                        algorithm=job.algorithm,
                        mhc_class=job.mhc_class,
                        peptide_length=job.peptide_length,
                        ic50=row.get("ic50", ""),
                        ic50_type="MHCnuggets mapped IC50(nM)",
                        percentile=percentile,
                        percentile_type="MHCnuggets human_proteome_percentile" if percentile else "",
                        raw_rank=rank,
                        raw_rank_type="MHCnuggets human_proteome_rank" if rank else "",
                        status="partial_ok" if rank_error else "ok",
                        raw_file=str(job.raw_path),
                        error=f"MHCnuggets rank output failed; IC50-only fallback used: {rank_error}"
                        if rank_error
                        else "",
                    )
                )
            for peptide in missing_peptides:
                predictions.append(
                    BindingPrediction(
                        peptide=peptide,
                        hla_allele=job.hla_allele,
                        algorithm=job.algorithm,
                        mhc_class=job.mhc_class,
                        peptide_length=job.peptide_length,
                        status="error",
                        raw_file=str(job.raw_path),
                        error="MHCnuggets did not return score for this peptide",
                    )
                )
            return write_ok_predictions(job, predictions)
        except Exception as exc:  # noqa: BLE001
            return write_error_predictions(job, exc)


def normalize_peptide(peptide: str) -> str:
    return str(peptide or "").strip().upper()


def build_mhcnuggets_command(job: PredictionJob, rank_output: bool, config) -> list[str]:
    cmd = [
        config.mhcnuggets_python_bin,
        config.mhcnuggets_script,
        "-c",
        CLASS_BY_ALGORITHM[job.algorithm],
        "-p",
        str(job.input_path),
        "-a",
        format_mhcnuggets_allele(job.hla_allele),
        "-o",
        str(job.raw_path),
    ]
    if rank_output:
        cmd.extend(["-r", "True"])
    return cmd


def format_mhcnuggets_allele(allele: str) -> str:
    """Convert common HLA notation to MHCnuggets model naming."""

    parts = []
    for part in allele.strip().split("/"):
        part = part.strip()
        if part.startswith("HLA-"):
            part = part[4:]
        parts.append("HLA-" + part.replace("*", ""))
    return "-".join(parts)


def read_mhcnuggets_rank_output(output_path: Path) -> dict[str, str]:
    """Read the optional MHCnuggets rank file matching an output path."""

    rank_path = first_existing_mhcnuggets_rank_path(output_path)
    if not rank_path.exists():
        return {}
    result: dict[str, str] = {}
    with rank_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            peptide = normalize_peptide(row.get("peptide", ""))
            if peptide:
                result[peptide] = row.get("human_proteome_rank", "")
    return result


def human_proteome_percentile(rank: str) -> str:
    """Convert MHCnuggets' 0-1 human-proteome rank to pVACtools' 0-100 scale."""

    if str(rank).strip() == "":
        return ""
    return f"{float(rank) * 100:.12g}"


def read_closest_mhcnuggets_allele(log_path: Path) -> str:
    """Read the trained allele selected by MHCnuggets' closest-model logic."""

    pattern = re.compile(r"^Closest allele found\s+(\S+)\s*$")
    for line in log_path.read_text().splitlines():
        match = pattern.match(line.strip())
        if match:
            return match.group(1)
    raise ValueError(f"MHCnuggets closest allele was not found in {log_path}")


def calculate_mhcnuggets_ranks(
    rows: list[dict[str, str]],
    mhc_class: str,
    predictor_allele: str,
    mhcnuggets_cwd: Path,
) -> dict[str, str]:
    """Calculate human-proteome ranks without upstream float-key lookups.

    MHCnuggets' bundled ``get_ranks`` indexes an exact float32 IC50 in a
    dictionary. Batch predictions can differ by floating-point representation
    and raise ``KeyError``. Bisecting the same reference distributions preserves
    the rank definition while making ties and rounded IC50 values robust.
    """

    suffix = "cI" if mhc_class == "MHC-I" else "cII"
    subdir = "mhcI" if mhc_class == "MHC-I" else "mhcII"
    data_dir = mhcnuggets_cwd / "mhcnuggets" / "data" / "production" / subdir
    ic50_data = read_pickle(data_dir / f"hp_ic50s_{suffix}.pkl")
    hp_lengths = read_pickle(data_dir / f"hp_ic50s_hp_lengths_{suffix}.pkl")
    first_percentiles = read_pickle(data_dir / f"hp_ic50s_first_percentiles_{suffix}.pkl")

    downsampled = list(ic50_data["downsampled"][predictor_allele])
    first_values = list(ic50_data["first_percentiles"][predictor_allele])
    first_percentile = float(first_percentiles[predictor_allele])
    hp_length = int(hp_lengths[predictor_allele])
    result = {}
    for row in rows:
        peptide = normalize_peptide(row.get("peptide", ""))
        ic50_text = str(row.get("ic50", "")).strip()
        if not peptide or not ic50_text:
            continue
        rank = reference_rank_fraction(
            float(ic50_text),
            first_percentile,
            downsampled,
            first_values,
            hp_length,
        )
        result[peptide] = f"{rank:.12g}"
    return result


def reference_rank_fraction(
    ic50: float,
    first_percentile: float,
    downsampled: list[float],
    first_values: list[float],
    hp_length: int,
) -> float:
    """Return the 0-1 MHCnuggets rank fraction using tie-aware bisect."""

    if ic50 > first_percentile:
        values = downsampled
        denominator = len(values)
    else:
        values = first_values
        denominator = hp_length
    if not values or denominator <= 0:
        raise ValueError("empty MHCnuggets human-proteome rank reference")
    left = bisect_left(values, ic50)
    right = bisect_right(values, ic50)
    position = (left + right - 1) / 2 if right > left else left - 1
    return max(0.0, min(1.0, (position + 1) / float(denominator)))


def read_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)  # noqa: S301 - trusted predictor installation data


def mhcnuggets_rank_path(output_path: Path) -> Path:
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}_ranks{output_path.suffix}")
    return Path(str(output_path) + "_ranks")


def mhcnuggets_actual_rank_path(output_path: Path) -> Path:
    """Return the rank filename produced by upstream MHCnuggets.

    The upstream script uses "{}_ranks.{}".format(file_name, extension), where
    extension already includes a leading dot. For an output named
    chunk.raw.tsv this yields chunk.raw_ranks..tsv.
    """

    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}_ranks.{output_path.suffix}")
    return Path(str(output_path) + "_ranks")


def first_existing_mhcnuggets_rank_path(output_path: Path) -> Path:
    for candidate in (mhcnuggets_rank_path(output_path), mhcnuggets_actual_rank_path(output_path)):
        if candidate.exists():
            return candidate
    return mhcnuggets_rank_path(output_path)
