import os
import re
import pandas as pd


STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
PROTEIN_HIT_QC_POLICY_VERSION = "microbial_protein_hit_qc_v1.0"

def parse_yp_taxids(qseqid: str):
    """
    Parse tax_ids from YP:Z: field in qseqid.
    Example qseqid:
      'B1|E2500...|99|150M|YP:Z:1912894,28116,371601,1912896'
    Returns:
      ['1912894', '28116', '371601', '1912896']
    """
    if not isinstance(qseqid, str):
        return []
    if "YP:Z:" not in qseqid:
        return []
    tail = qseqid.split("YP:Z:", 1)[1]
    tail = tail.split("|", 1)[0]
    return [x.strip() for x in tail.split(",") if x.strip()]


def filter_by_catalog_blastx(df_blast: pd.DataFrame, catalog_df: pd.DataFrame) -> pd.DataFrame:
    """
    Catalog filter for BLASTX results, analogous to DIAMOND filtering:
      - Normalize BLASTX sseqid (e.g. 'ref|WP_231809274.1|') to 'WP_231809274.1'
      - Parse YP tax_ids from qseqid
      - Expand to (qseqid, sseqid_clean, yp_taxid)
      - Inner-join with catalog (prot_id, tax_id)
      - Collapse back to rows that have at least one matched tax_id
      - Append matched tax_ids to qseqid as '|CTAXA:tax1,tax2,...'

    Returns a filtered DataFrame.
    """
    df = df_blast.copy()

    # catalog: prot_id, tax_id (tax_id as str)
    catalog_sub = catalog_df[["prot_id", "tax_id"]].copy()
    catalog_sub["tax_id"] = catalog_sub["tax_id"].astype(str)

    # normalize BLASTX sseqid, e.g. 'ref|WP_231809274.1|' -> 'WP_231809274.1'
    df["sseqid_clean"] = (
        df["sseqid"]
        .astype(str)
        .str.replace(r"^ref\|", "", regex=True)  # drop leading 'ref|'
        .str.rstrip("|")                         # drop trailing '|'
    )

    # parse YP taxids from qseqid
    df["yp_taxid_list"] = df["qseqid"].apply(parse_yp_taxids)

    # expand to long format: (qseqid, sseqid_clean, yp_taxid)
    tmp = df[["qseqid", "sseqid_clean", "yp_taxid_list"]].explode("yp_taxid_list")
    tmp = tmp.rename(columns={"yp_taxid_list": "yp_taxid"})
    tmp = tmp.dropna(subset=["yp_taxid"])

    if tmp.empty:
        return df.iloc[0:0].copy()

    # inner-join with catalog on (sseqid_clean, yp_taxid) ↔ (prot_id, tax_id)
    merged = pd.merge(
        tmp,
        catalog_sub,
        left_on=["sseqid_clean", "yp_taxid"],
        right_on=["prot_id", "tax_id"],
        how="inner",
    )
    if merged.empty:
        return df.iloc[0:0].copy()

    # collect matched tax_ids per (qseqid, sseqid_clean)
    matched = (
        merged.groupby(["qseqid", "sseqid_clean"])["yp_taxid"]
        .agg(lambda xs: sorted(set(xs)))
        .reset_index()
        .rename(columns={"yp_taxid": "matched_taxids"})
    )

    # merge back to original df (still keeps original sseqid for context)
    df_merged = df.merge(matched, on=["qseqid", "sseqid_clean"], how="left")
    df_filtered = df_merged[df_merged["matched_taxids"].notna()].copy()
    if df_filtered.empty:
        return df_filtered

    # build CTAXA suffix and append to qseqid
    df_filtered["matched_taxids_str"] = df_filtered["matched_taxids"].apply(
        lambda lst: ",".join(lst) if isinstance(lst, list) else str(lst)
    )

    # build CTAXA suffix and append to qseqid (add protein id too)
    df_filtered["qseqid"] = (
            df_filtered["qseqid"]
            + "|CTAXA:" + df_filtered["matched_taxids_str"]
            + "|PROT:" + df_filtered["sseqid_clean"].astype(str)
    )

    # drop intermediate columns
    df_filtered = df_filtered.drop(
        columns=["yp_taxid_list", "matched_taxids", "matched_taxids_str", "sseqid_clean"],
        errors="ignore",
    )

    return df_filtered


def extract_qseqid_tag(qseqid: str, tag: str) -> str:
    """Extract a pipe-delimited TAG:value field from qseqid."""
    if not isinstance(qseqid, str):
        return ""
    match = re.search(rf"(?:^|\|){re.escape(tag)}:([^|]+)", qseqid)
    return match.group(1) if match else ""


def normalize_parent_sequence(seq) -> tuple[str, list[str], bool]:
    """Normalize and validate a microbial parent amino-acid sequence."""
    if pd.isna(seq):
        return "", ["empty_parent_sequence"], False
    parent = str(seq).strip().upper()
    if not parent:
        return "", ["empty_parent_sequence"], False

    terminal_stop_removed = False
    if parent.endswith("*"):
        parent = parent[:-1]
        terminal_stop_removed = True

    reasons: list[str] = []
    if not parent:
        reasons.append("empty_after_terminal_stop_removal")
    if "*" in parent:
        reasons.append("internal_stop")
    if "-" in parent:
        reasons.append("gap_in_parent_sequence")

    noncanonical = sorted(set(parent).difference(STANDARD_AA))
    if noncanonical:
        reasons.append("noncanonical_amino_acid:" + ",".join(noncanonical))
    if len(parent) < 8:
        reasons.append("parent_length_lt_8")

    return parent, reasons, terminal_stop_removed


def append_reason(df: pd.DataFrame, mask: pd.Series, reason: str) -> None:
    """Append an exclusion reason to list-valued _qc_reasons cells."""
    for idx in df.index[mask.fillna(False)]:
        df.at[idx, "_qc_reasons"].append(reason)


def write_qc_summary(output_dir: str, sample: str | None, summary: dict) -> None:
    summary_path = os.path.join(
        output_dir,
        f"{sample if sample else 'blastx'}.protein_hits.qc_summary.tsv",
    )
    pd.DataFrame(
        [{"step": k, "n_hits": v} for k, v in summary.items()]
    ).to_csv(summary_path, sep="\t", index=False)

    legacy_summary_path = os.path.join(
        output_dir,
        f"{sample if sample else 'blastx'}.binding_pred_filter_stats.tsv",
    )
    pd.DataFrame(
        [{"step": k, "n_hits": v} for k, v in summary.items()]
    ).to_csv(legacy_summary_path, sep="\t", index=False)


def write_empty_stage1_outputs(output_dir: str, sample: str | None, pvacbind_file: str, summary: dict) -> None:
    sample_name = sample if sample else "blastx"
    open(os.path.join(output_dir, pvacbind_file), "w").close()
    pd.DataFrame().to_csv(os.path.join(output_dir, f"{sample_name}.blastx.filtered"), sep="\t", index=False)
    pd.DataFrame().to_csv(os.path.join(output_dir, f"{sample_name}.protein_hits.filtered.tsv"), sep="\t", index=False)
    pd.DataFrame().to_csv(os.path.join(output_dir, f"{sample_name}.protein_hits.excluded.tsv"), sep="\t", index=False)
    write_qc_summary(output_dir, sample, summary)


def build_standard_protein_hit_table(
    df_pass: pd.DataFrame,
    sample: str | None,
    policy_version: str = PROTEIN_HIT_QC_POLICY_VERSION,
) -> pd.DataFrame:
    """Build the neutral, search-engine-independent protein-hit table."""
    out = df_pass.copy()
    out["sample"] = sample if sample else ""
    out["source_row_number"] = out.get("_source_row_number", "")
    out["protein_hit_qc_policy_version"] = policy_version
    out["row_qc_status"] = "pass"
    out["row_qc_reasons"] = ""
    out["original_qseqid"] = out.get("original_qseqid", out["qseqid"])
    out["ctaxa_ids"] = out["qseqid"].map(lambda value: extract_qseqid_tag(value, "CTAXA"))
    out["protein_accession"] = out["qseqid"].map(lambda value: extract_qseqid_tag(value, "PROT"))
    out["parent_sequence_raw"] = out["sseq"].astype(str)
    return out


def build_excluded_table(
    df: pd.DataFrame,
    sample: str | None,
    policy_version: str = PROTEIN_HIT_QC_POLICY_VERSION,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["sample"] = sample if sample else ""
    out["source_row_number"] = out.get("_source_row_number", "")
    out["protein_hit_qc_policy_version"] = policy_version
    out["row_qc_status"] = "excluded"
    out["row_qc_reasons"] = out["_qc_reasons"].map(lambda values: ";".join(values))
    out["original_qseqid"] = out.get("original_qseqid", out["qseqid"])
    out["ctaxa_ids"] = out["qseqid"].map(lambda value: extract_qseqid_tag(value, "CTAXA"))
    out["protein_accession"] = out["qseqid"].map(lambda value: extract_qseqid_tag(value, "PROT"))
    out["parent_sequence_raw"] = out["sseq"].astype(str) if "sseq" in out.columns else ""
    return out.drop(columns=["_qc_reasons", "_source_row_number"], errors="ignore")



def get_data_for_binding_pred(
    blast_file: str,
    colnames: list,
    pvacbind_file: str,
    output_blastx: str,
    min_pident: float,
    catalog_df: pd.DataFrame,
    max_evalue: float = 1e-5,
    min_qcovs: float = 90.0,   # <-- renamed
    policy_version: str = PROTEIN_HIT_QC_POLICY_VERSION,
    sample=None,
    tool=None,
):
    """
    Process BLASTX results and generate peptide FASTA for binding prediction (pVACbind).

    Steps:
      1) Read BLASTX TSV file.
      2) Apply catalog filter (PathSeq YP tax_ids must be supported by V7_catalog_prot4).
      3) Filter by pident >= min_pident.
      4) Filter by evalue <= max_evalue.
      5) Filter by BLASTX qcovs >= min_qcovs (official subject coverage metric).
      6) Extract final peptides (sseq) into FASTA.
      7) Write summary table.

    Args:
        blast_file (str): BLASTX result filename.
        colnames (list): Column names from outfmt 6.
        pvacbind_file (str): Output FASTA file.
        output_blastx (str): Directory of BLASTX outputs.
        min_pident (float): Minimum identity.
        catalog_df (DataFrame): V7_catalog_prot4.
        max_evalue (float): E-value threshold.
        min_qcovs (float): Minimum qcovs threshold.
        sample (str|None): Sample ID.
        tool: Logger with write_log.

    Returns:
        dict: Summary statistics.
    """
    if tool is None:
        class DummyTool:
            def write_log(self, msg, level="info"):
                print(f"[{level.upper()}] {msg}")
        tool = DummyTool()

    blast_path = os.path.join(output_blastx, blast_file)
    if not os.path.exists(blast_path):
        tool.write_log(f"[{sample}] BLAST file not found: {blast_path}", "warning")
        return {}

    # Read BLASTX-compatible table
    df = pd.read_csv(blast_path, sep="\t", header=None, names=colnames)
    df["_source_row_number"] = range(1, len(df) + 1)
    df["original_qseqid"] = df["qseqid"] if "qseqid" in df.columns else ""

    summary = {}

    def _log_step(name, d):
        summary[name] = len(d)
        tool.write_log(f"[{sample}] {name}: {len(d)} hits", "info")

    # raw
    _log_step("raw_blastx", df)

    if df.empty:
        write_empty_stage1_outputs(output_blastx, sample, pvacbind_file, summary)
        return summary

    required_columns = {"qseqid", "sseqid", "sseq", "pident", "evalue", "qcovs"}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(
            f"[{sample}] BLASTX-compatible table missing required columns: {missing_columns}"
        )

    # convert numeric columns
    for col in ("pident", "evalue", "qcovs"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # step: after_taxonomic_consistency_check
    # BLASTX protein hits are retained only if their microbial taxon IDs
    # match PathSeq-derived YP taxon IDs at the species level, ensuring
    # species-level taxonomic consistency between nucleic and protein evidence.
    df_cat = filter_by_catalog_blastx(df, catalog_df)
    _log_step("after_taxonomic_consistency_check", df_cat)
    excluded_frames = []
    if "_source_row_number" in df_cat.columns:
        catalog_pass_rows = set(df_cat["_source_row_number"])
        df_catalog_excluded = df[~df["_source_row_number"].isin(catalog_pass_rows)].copy()
    else:
        df_catalog_excluded = df.iloc[0:0].copy()
    if not df_catalog_excluded.empty:
        df_catalog_excluded["_qc_reasons"] = [
            ["protein_taxon_catalog_mismatch"] for _ in range(len(df_catalog_excluded))
        ]
        excluded_frames.append(df_catalog_excluded)

    if df_cat.empty:
        excluded = pd.concat(excluded_frames, ignore_index=True) if excluded_frames else df.iloc[0:0].copy()
        build_excluded_table(excluded, sample, policy_version).to_csv(
            os.path.join(output_blastx, f"{sample if sample else 'blastx'}.protein_hits.excluded.tsv"),
            sep="\t",
            index=False,
        )
        open(os.path.join(output_blastx, pvacbind_file), "w").close()
        pd.DataFrame().to_csv(os.path.join(output_blastx, f"{sample if sample else 'blastx'}.blastx.filtered"), sep="\t", index=False)
        pd.DataFrame().to_csv(os.path.join(output_blastx, f"{sample if sample else 'blastx'}.protein_hits.filtered.tsv"), sep="\t", index=False)
        write_qc_summary(output_blastx, sample, summary)
        return summary

    df_work = df_cat.copy()
    df_work["_qc_reasons"] = [[] for _ in range(len(df_work))]
    df_work["pident"] = pd.to_numeric(df_work["pident"], errors="coerce")
    df_work["evalue"] = pd.to_numeric(df_work["evalue"], errors="coerce")
    df_work["qcovs"] = pd.to_numeric(df_work["qcovs"], errors="coerce")

    pident_pass = df_work["pident"] == float(min_pident)
    append_reason(df_work, ~pident_pass, f"pident_not_equal_{min_pident:g}")
    df_pid = df_work[pident_pass].copy()
    _log_step(f"after_pident_eq_{int(min_pident)}", df_pid)

    # evalue filtering
    evalue_pass = df_work["evalue"] <= float(max_evalue)
    append_reason(df_work, ~evalue_pass, f"evalue_gt_{max_evalue}")
    df_eval = df_work[pident_pass & evalue_pass].copy()
    _log_step(f"after_evalue_{max_evalue}", df_eval)

    # qcovs filtering
    qcovs_pass = df_work["qcovs"] >= float(min_qcovs)
    append_reason(df_work, ~qcovs_pass, f"qcovs_lt_{min_qcovs:g}")
    df_qcov = df_work[pident_pass & evalue_pass & qcovs_pass].copy()
    _log_step(f"after_qcovs_{int(min_qcovs)}", df_qcov)

    normalized = df_work["sseq"].map(normalize_parent_sequence)
    df_work["canonical_parent_sequence"] = [item[0] for item in normalized]
    df_work["terminal_stop_removed"] = [item[2] for item in normalized]
    canonical_reasons = [item[1] for item in normalized]
    for idx, reasons in zip(df_work.index, canonical_reasons):
        df_work.at[idx, "_qc_reasons"].extend(reasons)

    canonical_pass = df_work["_qc_reasons"].map(lambda values: len(values) == 0)
    df_pass = df_work[canonical_pass].copy()
    _log_step("after_canonical_parent_sequence", df_pass)

    excluded_filter = df_work[~canonical_pass].copy()
    if not excluded_filter.empty:
        excluded_frames.append(excluded_filter)
    excluded = pd.concat(excluded_frames, ignore_index=True) if excluded_frames else df_work.iloc[0:0].copy()

    excluded_path = os.path.join(output_blastx, f"{sample if sample else 'blastx'}.protein_hits.excluded.tsv")
    build_excluded_table(excluded, sample, policy_version).to_csv(excluded_path, sep="\t", index=False)

    # write summary
    write_qc_summary(output_blastx, sample, summary)

    if df_pass.empty:
        open(os.path.join(output_blastx, pvacbind_file), "w").close()
        pd.DataFrame().to_csv(os.path.join(output_blastx, f"{sample if sample else 'blastx'}.blastx.filtered"), sep="\t", index=False)
        pd.DataFrame().to_csv(os.path.join(output_blastx, f"{sample if sample else 'blastx'}.protein_hits.filtered.tsv"), sep="\t", index=False)
        return summary

    # ------------------------------------------------------------
    # Assign protIndex and write filtered protein-hit tables + FASTA
    # ------------------------------------------------------------
    # 1) Create a copy to avoid side-effects
    df_qcov_out = df_pass.copy().reset_index(drop=True)

    # 2) Add protIndex column (1-based) and rewrite qseqid as:
    #    protIndex:{i}|{original_qseqid_with_Ctaxa_and_Prot}
    df_qcov_out["protIndex"] = range(1, len(df_qcov_out) + 1)
    df_qcov_out["qseqid"] = df_qcov_out.apply(
        lambda r: f"protIndex:{int(r['protIndex'])}|{r['qseqid']}",
        axis=1
    )

    legacy_drop = [
        "_qc_reasons",
        "_source_row_number",
        "original_qseqid",
        "canonical_parent_sequence",
        "terminal_stop_removed",
    ]
    legacy_df = df_qcov_out.drop(columns=legacy_drop, errors="ignore")

    # 3) Save filtered BLASTX-compatible table for downstream compatibility
    filtered_path = os.path.join(output_blastx, f"{sample}.blastx.filtered")
    legacy_df.to_csv(filtered_path, sep="\t", index=False)
    tool.write_log(f"[{sample}] Wrote filtered BLASTX table: {filtered_path}", "info")

    standard_path = os.path.join(output_blastx, f"{sample}.protein_hits.filtered.tsv")
    build_standard_protein_hit_table(df_qcov_out, sample, policy_version).drop(
        columns=["_qc_reasons", "_source_row_number"], errors="ignore"
    ).to_csv(standard_path, sep="\t", index=False)
    tool.write_log(f"[{sample}] Wrote protein-hit QC table: {standard_path}", "info")

    # 4) Write FASTA using the SAME qseqid (already prefixed with protIndex)
    fasta_lines = []
    for row in df_qcov_out.itertuples(index=False):
        header = f">{row.qseqid}"
        peptide = str(row.canonical_parent_sequence)
        if not peptide:
            continue
        fasta_lines.append(header)
        fasta_lines.append(peptide)

    fasta_path = os.path.join(output_blastx, pvacbind_file)
    with open(fasta_path, "w") as f_out:
        f_out.write("\n".join(fasta_lines))

    tool.write_log(
        f"[{sample}] Wrote {len(fasta_lines) // 2} peptides to {fasta_path}",
        "info"
    )

    return summary
