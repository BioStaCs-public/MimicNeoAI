import hashlib
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from mimicneoai.functions.nodemon_pool import NoDaemonPool
from mimicneoai.immunogenicity_prediction.model import (
    Vocab,
    build_model,
    checkpoint_payload,
    load_model_weights,
)
from mimicneoai.immunogenicity_prediction.runtime.hla import normalize_hla
from mimicneoai.immunogenicity_prediction.runtime.qc import (
    INPUT_QC_COLUMNS,
    annotate_input_qc,
)


AMINO = "ARNDCQEGHILKMFPSTWYV-"
PADDING_LENGTH = 419
X2_DIM = 25
SEPARATOR_LENGTH = 12
PHYSICOCHEMICAL_FEATURE_COLUMNS = (
    "AA Composition",
    "Polarity",
    "Volume",
    "Net Charge",
    "Hydrophobicity",
    "Boman Index",
    "Aliphatic Index",
    "Isoelectric Point",
)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HLA_PSEUDOSEQ_DIR = SCRIPT_DIR / "resources" / "hla_pseudoseq" / "local"
DEFAULT_HLA_CLASS1_PSEUDOSEQ = DEFAULT_HLA_PSEUDOSEQ_DIR / "netmhcpan_class1_allele_to_pseudoseq.csv"
DEFAULT_HLA_CLASS2_PSEUDOSEQ = DEFAULT_HLA_PSEUDOSEQ_DIR / "netmhciipan_class2_allele_to_pseudoseq.csv"
DEFAULT_FEATURE_RSCRIPT = SCRIPT_DIR / "compute_peptide_features.R"

@dataclass
class InferenceConfig:
    model_path: str
    hla_fasta_path: str = ""
    hla_source: str = "fasta"
    hla_pseudoseq_csv: Tuple[str, ...] = ()
    padding_length: int = 0
    peptide_col: str = "peptide"
    hla_col: str = "hla"
    source_col: str = "antigen_class"
    output_score_col: str = "immunogenicity_score"
    output_status_col: str = "immunogenicity_status"
    batch_size: int = 512
    device: str = "auto"
    num_processes: int = 1
    verbose: bool = True
    include_input_qc: bool = False


def pick_device(device_name: str) -> torch.device:
    name = str(device_name).strip().lower()
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda:0")
    if name.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"{device_name} requested but CUDA is not available.")
        idx = int(name.split(":", 1)[1])
        if idx < 0 or idx >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device index out of range: {idx}. "
                f"Available count: {torch.cuda.device_count()}."
            )
        return torch.device(f"cuda:{idx}")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parse_hla_fasta(hla_fasta_path: str) -> Dict[str, str]:
    hla_names: List[str] = []
    prots: List[str] = []

    with open(hla_fasta_path, "r", encoding="utf-8") as f:
        prot = ""
        started = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                tokens = line.split()
                chosen = None
                for tok in tokens[1:]:
                    clean = tok.replace("HLA-", "")
                    if "*" in clean:
                        chosen = clean
                        break
                if chosen is None:
                    chosen = tokens[0].replace(">", "")
                hla_names.append(chosen)
                if started:
                    prots.append(prot)
                    prot = ""
                started = True
            else:
                prot += line

    if started:
        prots.append(prot)

    hla_df = pd.DataFrame({"HLA": hla_names, "protin": prots})
    hla_df["HLA"] = hla_df["HLA"].astype(str).str.split(":").str[:2].str.join(":")
    hla_df = hla_df.drop_duplicates(subset="HLA", keep="first").reset_index(drop=True)
    return dict(zip(hla_df["HLA"], hla_df["protin"]))


def parse_hla_pseudoseq_csv(paths: Sequence[str | Path]) -> Dict[str, str]:
    hla2seq: Dict[str, str] = {}
    path_objs = [Path(path) for path in paths]
    missing = [str(path) for path in path_objs if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing HLA pseudo-sequence CSV(s): "
            + ", ".join(missing)
            + ". Place local NetMHC-derived resources under "
            + str(DEFAULT_HLA_PSEUDOSEQ_DIR)
            + " or configure hla_pseudoseq_csv."
        )
    for path in path_objs:
        df = pd.read_csv(path)
        required = {"allele", "pseudo_sequence"}
        if not required.issubset(df.columns):
            raise KeyError(f"{path} must contain columns: {sorted(required)}")
        for row in df.itertuples(index=False):
            allele = str(getattr(row, "allele")).strip()
            seq = str(getattr(row, "pseudo_sequence")).strip().upper()
            if not allele or not seq or seq.lower() == "nan":
                continue
            keys = {
                allele,
                allele.replace("HLA-", ""),
                normalize_hla(allele),
            }
            for key in keys:
                if key:
                    hla2seq[key] = seq
    return hla2seq


def load_hla_sequences(cfg: InferenceConfig) -> Dict[str, str]:
    if cfg.hla_source == "fasta":
        if not cfg.hla_fasta_path:
            raise ValueError("hla_fasta_path is required when hla_source='fasta'.")
        return parse_hla_fasta(cfg.hla_fasta_path)
    if cfg.hla_source == "netmhc-pseudoseq":
        paths = cfg.hla_pseudoseq_csv or (
            str(DEFAULT_HLA_CLASS1_PSEUDOSEQ),
            str(DEFAULT_HLA_CLASS2_PSEUDOSEQ),
        )
        return parse_hla_pseudoseq_csv(paths)
    raise ValueError(f"Unsupported HLA source: {cfg.hla_source}")


def calculate_peptide_features(df: pd.DataFrame, peptide_col: str) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="mimicneoai_features_") as tmp_dir:
        input_path = Path(tmp_dir) / "input.csv"
        output_path = Path(tmp_dir) / "output.csv"
        df.to_csv(input_path, index=False)
        subprocess.run(
            [
                "Rscript",
                str(DEFAULT_FEATURE_RSCRIPT),
                str(input_path),
                str(output_path),
                peptide_col,
            ],
            check=True,
        )
        return pd.read_csv(output_path)


def _calc_features_worker(args):
    chunk, peptide_col = args
    return calculate_peptide_features(chunk, peptide_col)


def _split_dataframe(df: pd.DataFrame, n_chunks: int) -> List[pd.DataFrame]:
    n = len(df)
    if n == 0:
        return []
    n_chunks = max(1, min(int(n_chunks), n))
    base = n // n_chunks
    rem = n % n_chunks
    chunks: List[pd.DataFrame] = []
    start = 0
    for i in range(n_chunks):
        end = start + base + (1 if i < rem else 0)
        chunk = df.iloc[start:end]
        if len(chunk) > 0:
            chunks.append(chunk.copy())
        start = end
    return chunks


def parallel_calculate_features(df: pd.DataFrame, peptide_col: str, num_processes: int) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()

    n = max(1, min(int(num_processes), len(df)))
    if n == 1:
        return calculate_peptide_features(df, peptide_col)

    chunks = _split_dataframe(df, n)
    with NoDaemonPool(processes=len(chunks)) as pool:
        iterator = pool.imap(_calc_features_worker, [(c, peptide_col) for c in chunks])
        results = list(tqdm(iterator, total=len(chunks), desc="Feature chunks", leave=True))

    return pd.concat(results, ignore_index=True)


def build_feature_table(input_df: pd.DataFrame, peptide_col: str, num_processes: int) -> pd.DataFrame:
    unique_peptides = input_df[[peptide_col]].drop_duplicates().reset_index(drop=True)
    feat = parallel_calculate_features(unique_peptides, peptide_col, num_processes)
    feat = feat.rename(
        columns={
            "aa_composition": "AA Composition",
            "polarity": "Polarity",
            "volume": "Volume",
            "net_charge": "Net Charge",
            "hydrophobicity": "Hydrophobicity",
            "boman_index": "Boman Index",
            "aliphatic_index": "Aliphatic Index",
            "isoelectric_point": "Isoelectric Point",
        }
    )
    return input_df.merge(feat, on=peptide_col, how="left")


def aaindex_vocab(seq: str, vocab_map: Dict[str, int]) -> np.ndarray:
    encoded = np.empty([len(seq)], dtype=np.float32)
    for i, aa in enumerate(seq):
        query = "-" if aa.upper() == "X" else aa.upper()
        encoded[i] = vocab_map.get(query, 0)
    return encoded


def _feature_matrix_chunk_worker(chunk: pd.DataFrame) -> List[List[float]]:
    chunk = chunk.reset_index(drop=True)
    rows: List[List[float]] = []
    for i in range(len(chunk)):
        aa_comp = [float(x.strip()) for x in str(chunk.iloc[i]["AA Composition"]).split(",")]
        other_feats = chunk.iloc[i][
            ["Polarity", "Volume", "Net Charge", "Hydrophobicity", "Boman Index", "Aliphatic Index", "Isoelectric Point"]
        ].tolist()
        rows.append(aa_comp + [float(x) for x in other_feats])
    return rows


def extract_feature_vector(df: pd.DataFrame, num_processes: int) -> np.ndarray:
    n = len(df)
    if n == 0:
        return np.empty((0, X2_DIM), dtype=np.float32)

    p = max(1, min(int(num_processes), n))
    if p == 1:
        all_features = _feature_matrix_chunk_worker(df)
        return np.array(all_features, dtype=np.float32)

    chunks = _split_dataframe(df, p)
    with NoDaemonPool(processes=len(chunks)) as pool:
        iterator = pool.imap(_feature_matrix_chunk_worker, chunks)
        chunk_rows = list(tqdm(iterator, total=len(chunks), desc="Build feature matrix", leave=True))

    all_features = [row for rows in chunk_rows for row in rows]
    return np.array(all_features, dtype=np.float32)


def _encode_chunk_worker(args):
    features_chunk, peptides_chunk, hlas_chunk, hla2prot, vocab_map, padding_length, idx_chunk = args
    encoded_batch = []
    flags = []
    out_idx = []

    for i in range(len(peptides_chunk)):
        peptide = peptides_chunk[i]
        hla = hlas_chunk[i]
        hla_paratope = hla2prot.get(hla)

        if hla_paratope is None:
            encoded_batch.append(np.zeros(padding_length, dtype=np.float32))
            flags.append(0)
            out_idx.append(idx_chunk[i])
            continue

        peptide_encode = aaindex_vocab(peptide, vocab_map)
        hla_encode = aaindex_vocab(hla_paratope, vocab_map)
        feature_array = np.asarray(features_chunk[i], dtype=np.float32)

        zero_array = np.zeros(SEPARATOR_LENGTH, dtype=np.float32)
        merged = np.concatenate((feature_array, peptide_encode, zero_array, hla_encode), axis=0)

        if merged.shape[0] > padding_length:
            merged = merged[:padding_length]

        pad_len = padding_length - merged.shape[0]
        merged_padded = np.pad(merged, (0, pad_len), "constant")
        encoded_batch.append(merged_padded.astype(np.float32))
        flags.append(1)
        out_idx.append(idx_chunk[i])

    return encoded_batch, flags, out_idx


def parallel_encode(
    features_np: np.ndarray,
    peptides: List[str],
    hlas: List[str],
    hla2prot: Dict[str, str],
    vocab_map: Dict[str, int],
    padding_length: int,
    num_processes: int,
) -> Tuple[torch.Tensor, List[int]]:
    n = len(peptides)
    if n == 0:
        return torch.empty((0, padding_length), dtype=torch.float32), []

    p = max(1, min(int(num_processes), n))
    if p == 1:
        enc, flags, _ = _encode_chunk_worker(
            (features_np, peptides, hlas, hla2prot, vocab_map, padding_length, list(range(n)))
        )
        return torch.from_numpy(np.stack(enc, axis=0)), flags

    chunk_size = n // p + int(n % p > 0)
    args = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        args.append(
            (
                features_np[start:end],
                peptides[start:end],
                hlas[start:end],
                hla2prot,
                vocab_map,
                padding_length,
                list(range(start, end)),
            )
        )

    with NoDaemonPool(processes=len(args)) as pool:
        iterator = pool.imap(_encode_chunk_worker, args)
        results = list(tqdm(iterator, total=len(args), desc="Encoding chunks", leave=True))

    encoded_arrays = [arr for (enc, _, _) in results for arr in enc]
    flags = [f for (_, flg, _) in results for f in flg]
    row_idx = [ix for (_, _, idxs) in results for ix in idxs]

    if row_idx != list(range(n)):
        reorder = np.argsort(np.array(row_idx))
        encoded_arrays = [encoded_arrays[i] for i in reorder]
        flags = [flags[i] for i in reorder]

    encoded_tensor = torch.from_numpy(np.stack(encoded_arrays, axis=0))
    return encoded_tensor, flags


def resolve_padding_length(
    requested_padding_length: int,
    hla_source: str,
    peptides: Sequence[str],
    hla2prot: Dict[str, str],
) -> int:
    if requested_padding_length > 0:
        return int(requested_padding_length)
    if hla_source == "fasta":
        return PADDING_LENGTH
    max_peptide_len = max((len(str(peptide)) for peptide in peptides), default=0)
    max_hla_len = max((len(seq) for seq in hla2prot.values()), default=0)
    return X2_DIM + max_peptide_len + SEPARATOR_LENGTH + max_hla_len


def prepare_inference_features(input_df: pd.DataFrame, cfg: InferenceConfig) -> pd.DataFrame:
    """Normalize peptide-HLA input and calculate its reusable 25D feature table."""
    if cfg.peptide_col not in input_df.columns or cfg.hla_col not in input_df.columns:
        raise KeyError(f"Input must contain columns '{cfg.peptide_col}' and '{cfg.hla_col}'.")
    work_df = (
        annotate_input_qc(input_df, peptide_col=cfg.peptide_col, hla_col=cfg.hla_col)
        if cfg.include_input_qc
        else input_df.copy()
    )
    work_df[cfg.peptide_col] = work_df[cfg.peptide_col].astype(str).str.strip().str.upper()
    work_df[cfg.hla_col] = work_df[cfg.hla_col].astype(str).str.strip()
    work_df["_norm_hla"] = (
        work_df["input_qc_normalized_hla"]
        if cfg.include_input_qc
        else work_df[cfg.hla_col].apply(normalize_hla)
    )

    return build_feature_table(work_df, cfg.peptide_col, cfg.num_processes)


def run_inference(
    input_df: pd.DataFrame,
    cfg: InferenceConfig,
    *,
    prepared_feature_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if cfg.peptide_col not in input_df.columns or cfg.hla_col not in input_df.columns:
        raise KeyError(f"Input must contain columns '{cfg.peptide_col}' and '{cfg.hla_col}'.")

    t0 = time.time()
    if cfg.verbose:
        print("[Immunogenicity] Start inference")
        print(
            f"[Immunogenicity] rows={len(input_df)} "
            f"batch_size={cfg.batch_size} num_processes={cfg.num_processes} device={cfg.device}"
        )

    t_feat = time.time()
    if prepared_feature_table is None:
        if cfg.verbose:
            print("[Immunogenicity] Extracting peptide physicochemical features...")
        work_df = prepare_inference_features(input_df, cfg)
        if cfg.verbose:
            print(f"[Immunogenicity] Feature extraction done in {time.time() - t_feat:.2f}s")
    else:
        required_features = {cfg.peptide_col, cfg.hla_col, "_norm_hla", *PHYSICOCHEMICAL_FEATURE_COLUMNS}
        if cfg.include_input_qc:
            required_features.update(INPUT_QC_COLUMNS)
        missing_features = required_features.difference(prepared_feature_table.columns)
        if missing_features:
            raise KeyError(
                "Prepared feature table is missing columns: "
                f"{sorted(missing_features)}"
            )
        if len(prepared_feature_table) != len(input_df):
            raise ValueError("Prepared feature table row count does not match input_df.")
        work_df = prepared_feature_table.copy()
        if cfg.verbose:
            print(f"[Immunogenicity] Reusing peptide physicochemical features ({time.time() - t_feat:.2f}s)")

    infer_columns = [
            cfg.peptide_col,
            cfg.hla_col,
            *PHYSICOCHEMICAL_FEATURE_COLUMNS,
    ]
    if cfg.source_col in work_df.columns:
        infer_columns.append(cfg.source_col)
    infer_df = work_df[infer_columns].copy()

    infer_df["_norm_hla"] = infer_df[cfg.hla_col].apply(normalize_hla)
    dedup_columns = ["_norm_hla", cfg.peptide_col]
    if cfg.source_col in infer_df.columns:
        dedup_columns.append(cfg.source_col)
    infer_df = infer_df.drop_duplicates(subset=dedup_columns).reset_index(drop=True)

    device = pick_device(cfg.device)
    vocab = Vocab(list(AMINO), min_freq=1)
    hla2prot = load_hla_sequences(cfg)

    features_np = extract_feature_vector(infer_df, cfg.num_processes)
    peptides = infer_df[cfg.peptide_col].tolist()
    hlas = infer_df["_norm_hla"].tolist()
    payload = checkpoint_payload(cfg.model_path, device)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    model_config = metadata.get("model_config", {})
    source_mapping = metadata.get("source_mapping", {})
    checkpoint_hla_source = metadata.get("hla_source")
    if checkpoint_hla_source and checkpoint_hla_source != cfg.hla_source:
        raise ValueError(
            f"Inference hla_source={cfg.hla_source} does not match checkpoint hla_source={checkpoint_hla_source}"
        )
    if model_config.get("architecture") == "source_aware" and cfg.source_col not in infer_df.columns:
        raise KeyError(f"Source-aware checkpoint requires input column '{cfg.source_col}'")
    checkpoint_padding = int(metadata.get("padding_length", 0) or 0)
    padding_length = resolve_padding_length(
        cfg.padding_length or checkpoint_padding,
        cfg.hla_source,
        peptides,
        hla2prot,
    )

    scaler = metadata.get("physchem_scaler")
    if scaler is not None:
        expected_digest = metadata.get("physchem_scaler_sha256")
        actual_digest = hashlib.sha256(json.dumps(scaler, sort_keys=True).encode("utf-8")).hexdigest()
        if expected_digest and actual_digest != expected_digest:
            raise RuntimeError("Checkpoint-bound physicochemical scaler digest mismatch")
        mean = np.asarray(scaler["mean"], dtype=np.float32)
        scale = np.asarray(scaler["scale"], dtype=np.float32)
        features_np = (features_np - mean) / scale
        clip = float(scaler.get("clip", 0.0))
        if clip > 0:
            features_np = np.clip(features_np, -clip, clip)
        if not np.isfinite(features_np).all():
            raise RuntimeError("Checkpoint-bound physicochemical scaling produced NaN/Inf")
    elif model_config.get("physchem_fusion") == "embedding_refine":
        raise RuntimeError("embedding_refine checkpoint is missing its bound physicochemical scaler")

    t_enc = time.time()
    if cfg.verbose:
        print(
            "[Immunogenicity] Encoding peptide/HLA pairs... "
            f"hla_source={cfg.hla_source} padding_length={padding_length}"
        )
    encoded_all, flags_all = parallel_encode(
        features_np=features_np,
        peptides=peptides,
        hlas=hlas,
        hla2prot=hla2prot,
        vocab_map=vocab.token_to_idx,
        padding_length=padding_length,
        num_processes=cfg.num_processes,
    )
    if cfg.verbose:
        print(f"[Immunogenicity] Encoding done in {time.time() - t_enc:.2f}s")
        missing_hla_count = len(flags_all) - int(sum(flags_all))
        if missing_hla_count:
            print(f"[Immunogenicity] Missing HLA sequence rows: {missing_hla_count}")

    model = build_model(vocab_size=len(vocab), pad_idx=vocab["<pad>"], model_config=model_config)
    load_model_weights(model, cfg.model_path, device)
    model.to(device)
    model.eval()

    probs_out: List[float] = []
    status_out: List[str] = []
    missing_hla_status = (
        "missing_hla_pseudoseq" if cfg.hla_source == "netmhc-pseudoseq" else "missing_hla_fasta"
    )
    t_inf = time.time()
    if cfg.verbose:
        print("[Immunogenicity] Running model inference...")
    with torch.no_grad():
        for start in tqdm(range(0, len(infer_df), cfg.batch_size), desc="Infer batches", leave=True):
            end = min(start + cfg.batch_size, len(infer_df))
            batch_encoded = encoded_all[start:end].to(device)
            batch_flags = flags_all[start:end]

            x1 = batch_encoded[:, X2_DIM:].long()
            x2 = batch_encoded[:, 0:X2_DIM].float()
            source_ids = None
            if model_config.get("architecture") == "source_aware":
                source_values = infer_df.iloc[start:end][cfg.source_col].astype(str).tolist()
                missing_sources = sorted(set(source_values) - set(source_mapping))
                if missing_sources:
                    raise KeyError(f"Checkpoint source mapping does not contain: {missing_sources}")
                source_ids = torch.tensor(
                    [source_mapping[value] for value in source_values],
                    dtype=torch.long,
                    device=device,
                )
            if model_config.get("physchem_fusion") == "embedding_refine":
                lengths = torch.tensor(
                    [len(peptide) for peptide in peptides[start:end]],
                    dtype=torch.long,
                    device=device,
                )
                present = torch.ones(len(lengths), dtype=torch.float32, device=device)
                if source_ids is None:
                    logits = model(x1, x2, peptide_lengths=lengths, physchem_present=present)
                else:
                    logits = model(
                        x1,
                        x2,
                        source_ids,
                        peptide_lengths=lengths,
                        physchem_present=present,
                    )
            else:
                logits = model(x1, x2) if source_ids is None else model(x1, x2, source_ids)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            for j, flag in enumerate(batch_flags):
                if flag:
                    probs_out.append(float(probs[j, 1]))
                    status_out.append("ok")
                else:
                    probs_out.append(float("nan"))
                    status_out.append(missing_hla_status)

    infer_df[cfg.output_score_col] = probs_out
    infer_df[cfg.output_status_col] = status_out

    merge_keys = [cfg.peptide_col, "_norm_hla"]
    if model_config.get("architecture") == "source_aware":
        merge_keys.append(cfg.source_col)
    scored = work_df.merge(
        infer_df[merge_keys + [cfg.output_score_col, cfg.output_status_col]],
        on=merge_keys,
        how="left",
    )
    if cfg.include_input_qc:
        scored["input_qc_hla_reference_status"] = np.where(
            scored[cfg.output_status_col].eq("ok"),
            "matched",
            scored[cfg.output_status_col],
        )
        missing_reference = ~scored[cfg.output_status_col].eq("ok")
        if missing_reference.any():
            additions = scored.loc[missing_reference, "input_qc_hla_reference_status"]
            existing = scored.loc[missing_reference, "input_qc_flags"].fillna("")
            scored.loc[missing_reference, "input_qc_flags"] = [
                f"{flag};{addition}" if flag else str(addition)
                for flag, addition in zip(existing, additions)
            ]
            scored.loc[missing_reference, "input_qc_status"] = "warning"
    scored = scored.drop(columns=["_norm_hla"])

    if cfg.verbose:
        print(f"[Immunogenicity] Inference done in {time.time() - t_inf:.2f}s")
        print(f"[Immunogenicity] Total time {time.time() - t0:.2f}s")

    return scored


def export_model_to_onnx(
    model_path: str,
    onnx_path: str,
    device_name: str = "cpu",
    opset_version: int = 17,
    padding_length: int = PADDING_LENGTH,
) -> None:
    device = pick_device(device_name)
    vocab = Vocab(list(AMINO), min_freq=1)
    payload = checkpoint_payload(model_path, device)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    model_config = metadata.get("model_config", {})
    model = build_model(vocab_size=len(vocab), pad_idx=vocab["<pad>"], model_config=model_config)
    load_model_weights(model, model_path, device)
    model.to(device)
    model.eval()

    x1_len = int(padding_length) - X2_DIM
    dummy_x1 = torch.zeros((1, x1_len), dtype=torch.long, device=device)
    dummy_x2 = torch.zeros((1, X2_DIM), dtype=torch.float32, device=device)

    embedding_refine = model_config.get("physchem_fusion") == "embedding_refine"
    source_aware = model_config.get("architecture") == "source_aware"
    dummy_source = torch.zeros((1,), dtype=torch.long, device=device)
    if embedding_refine:
        dummy_lengths = torch.full((1,), 9, dtype=torch.long, device=device)
        dummy_present = torch.ones((1,), dtype=torch.float32, device=device)
        if source_aware:
            model_inputs = (dummy_x1, dummy_x2, dummy_source, dummy_lengths, dummy_present)
            input_names = ["x1", "x2", "source_id", "peptide_lengths", "physchem_present"]
        else:
            model_inputs = (dummy_x1, dummy_x2, dummy_lengths, dummy_present)
            input_names = ["x1", "x2", "peptide_lengths", "physchem_present"]
        dynamic_axes = {
            "x1": {0: "batch", 1: "seq_len"},
            "x2": {0: "batch"},
            "peptide_lengths": {0: "batch"},
            "physchem_present": {0: "batch"},
            "logits": {0: "batch"},
        }
        if source_aware:
            dynamic_axes["source_id"] = {0: "batch"}
    else:
        model_inputs = (dummy_x1, dummy_x2, dummy_source) if source_aware else (dummy_x1, dummy_x2)
        input_names = ["x1", "x2", "source_id"] if source_aware else ["x1", "x2"]
        dynamic_axes = {
            "x1": {0: "batch", 1: "seq_len"},
            "x2": {0: "batch"},
            "logits": {0: "batch"},
        }
        if source_aware:
            dynamic_axes["source_id"] = {0: "batch"}
    torch.onnx.export(
        model,
        model_inputs,
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )
