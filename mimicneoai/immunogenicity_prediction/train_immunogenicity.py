#!/usr/bin/env python3
"""Train MimicNeoAI immunogenicity models from CSV inputs.

This script keeps the original notebook data representation and model path:
25 peptide physicochemical features + peptide AAIndex/PCA indices + 12 zeros
+ HLA AAIndex/PCA indices. By default this reproduces the original full-length
HLA input padded to length 419. NetMHC pseudo-sequence HLA inputs can be used
with --hla-source netmhc-pseudoseq.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch import nn
from torch.utils import data

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else []


AMINO = "ARNDCQEGHILKMFPSTWYV-"
PADDING_LENGTH = 419
X2_DIM = 25
SEPARATOR_LENGTH = 12
FEATURE_COLUMNS = [
    "aa_composition",
    "polarity",
    "volume",
    "net_charge",
    "hydrophobicity",
    "boman_index",
    "aliphatic_index",
    "isoelectric_point",
]
ENCODING_CACHE_VERSION = "peptide_hla_tensor_v2_explicit_lengths"
ZERO_AA_COMPOSITION = ", ".join(["0"] * 18)


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent
DEFAULT_MODEL_FILE = SCRIPT_DIR / "model.py"
DEFAULT_FEATURE_RSCRIPT = SCRIPT_DIR / "compute_peptide_features.R"
DEFAULT_AFTER_PCA = SCRIPT_DIR / "resources" / "after_pca.txt"
DEFAULT_HLA_FASTA = PACKAGE_ROOT / "example" / "immunogenicity_prediction" / "models" / "hla_prot.fasta"
DEFAULT_HLA_PSEUDOSEQ_DIR = SCRIPT_DIR / "resources" / "hla_pseudoseq" / "local"
DEFAULT_HLA_CLASS1_PSEUDOSEQ = DEFAULT_HLA_PSEUDOSEQ_DIR / "netmhcpan_class1_allele_to_pseudoseq.csv"
DEFAULT_HLA_CLASS2_PSEUDOSEQ = DEFAULT_HLA_PSEUDOSEQ_DIR / "netmhciipan_class2_allele_to_pseudoseq.csv"


def log_step(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


class Vocab:
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = collections.Counter()
        for token in tokens:
            if isinstance(token, list):
                counter.update(token)
            else:
                counter[token] += 1
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ["<pad>"] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self.token_freqs:
            if freq >= min_freq and token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, 0)
        return [self.__getitem__(token) for token in tokens]


def load_model_class(model_file: Path, architecture: str):
    spec = importlib.util.spec_from_file_location("mimicneoai_model", model_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import model file: {model_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if architecture == "phla":
        return module.NewBiLSTM
    if architecture == "source_aware":
        if not hasattr(module, "SourceAwareBiLSTM"):
            raise AttributeError(f"{model_file} does not define SourceAwareBiLSTM")
        return module.SourceAwareBiLSTM
    raise ValueError(f"Unsupported architecture: {architecture}")


def read_table_auto(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t", low_memory=False)
    return pd.read_csv(path, low_memory=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def normalize_hla(hla: str) -> str:
    val = str(hla).strip()
    val = re.sub(r"^HLA-", "", val, flags=re.IGNORECASE)
    if "-" in val:
        val = re.sub(r"(?<=[0-9])-(?=[A-Z])", "/", val)
    if "/" in val:
        val = val.split("/")[-1]
    parts = val.split(":")
    if len(parts) >= 2:
        val = f"{parts[0]}:{parts[1]}"
    return val


def parse_hla_fasta(hla_fasta_path: Path) -> Dict[str, str]:
    hla_names: List[str] = []
    prots: List[str] = []
    prot = ""
    started = False
    with hla_fasta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
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


def parse_hla_pseudoseq_csv(paths: Sequence[Path]) -> Dict[str, str]:
    hla2seq: Dict[str, str] = {}
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing HLA pseudo-sequence CSV(s): "
            + ", ".join(missing)
            + ". Place local NetMHC-derived resources under "
            + str(DEFAULT_HLA_PSEUDOSEQ_DIR)
            + " or pass --hla-pseudoseq-csv."
        )
    for path in paths:
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


def load_hla_sequences(
    source: str,
    hla_fasta: Path,
    hla_pseudoseq_csv: Sequence[Path] | None,
) -> Dict[str, str]:
    if source == "fasta":
        return parse_hla_fasta(hla_fasta)
    if source == "netmhc-pseudoseq":
        paths = list(hla_pseudoseq_csv or [DEFAULT_HLA_CLASS1_PSEUDOSEQ, DEFAULT_HLA_CLASS2_PSEUDOSEQ])
        return parse_hla_pseudoseq_csv(paths)
    raise ValueError(f"Unsupported HLA source: {source}")


def standardize_input(df: pd.DataFrame, source: str) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        low = col.strip().lower()
        if low == "peptide":
            rename[col] = "peptide"
        elif low == "hla":
            rename[col] = "hla"
        elif low == "label":
            rename[col] = "label"
        elif low == "length":
            rename[col] = "length"
    df = df.rename(columns=rename).copy()
    missing = {"peptide", "hla", "label"} - set(df.columns)
    if missing:
        raise KeyError(f"{source} missing required columns: {sorted(missing)}")
    df["peptide"] = df["peptide"].astype(str).str.strip().str.upper()
    df["hla"] = df["hla"].astype(str).str.strip()
    df["_norm_hla"] = df["hla"].map(normalize_hla)
    df["label"] = df["label"].astype(int)
    return df


def sha_for_peptides(peptides: Sequence[str]) -> str:
    h = hashlib.sha256()
    for pep in sorted(set(peptides)):
        h.update(pep.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def sha_for_hla_sequences(hla2seq: Dict[str, str]) -> str:
    h = hashlib.sha256()
    for allele, seq in sorted(hla2seq.items()):
        h.update(str(allele).encode("utf-8"))
        h.update(b"\t")
        h.update(str(seq).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def sha_for_dataframe_columns(df: pd.DataFrame, columns: Sequence[str]) -> str:
    h = hashlib.sha256()
    subset = df.loc[:, list(columns)].copy()
    h.update("\t".join(columns).encode("utf-8"))
    h.update(b"\n")
    h.update(str(len(subset)).encode("utf-8"))
    h.update(b"\n")
    row_hashes = pd.util.hash_pandas_object(subset, index=False).to_numpy(dtype=np.uint64)
    h.update(row_hashes.tobytes())
    return h.hexdigest()[:16]


def ensure_peptide_features(
    df: pd.DataFrame,
    cache_dir: Path,
    rscript_path: Path,
    peptide_col: str = "peptide",
    feature_workers: int = 1,
    feature_chunk_size: int = 250000,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    unique = pd.DataFrame({peptide_col: sorted(df[peptide_col].dropna().astype(str).unique())})
    cache_key = sha_for_peptides(unique[peptide_col].tolist())
    out_csv = cache_dir / f"peptide_features_{cache_key}.csv"
    in_csv = cache_dir / f"peptide_features_{cache_key}.input.csv"
    if not out_csv.exists():
        unique.to_csv(in_csv, index=False)
        t0 = time.time()
        if feature_workers <= 1 or len(unique) <= feature_chunk_size:
            cmd = ["Rscript", str(rscript_path), str(in_csv), str(out_csv), peptide_col]
            log_step(f"[features] running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        else:
            feature_workers = max(1, int(feature_workers))
            feature_chunk_size = max(1, int(feature_chunk_size))
            n_chunks = math.ceil(len(unique) / feature_chunk_size)
            chunks_dir = cache_dir / f"peptide_features_{cache_key}.chunks"
            chunks_dir.mkdir(parents=True, exist_ok=True)
            jobs = []
            for chunk_idx in range(n_chunks):
                start = chunk_idx * feature_chunk_size
                end = min(start + feature_chunk_size, len(unique))
                chunk = unique.iloc[start:end].copy()
                chunk_in = chunks_dir / f"chunk_{chunk_idx:04d}.input.csv"
                chunk_out = chunks_dir / f"chunk_{chunk_idx:04d}.features.csv"
                chunk.to_csv(chunk_in, index=False)
                jobs.append((chunk_idx, chunk_in, chunk_out, len(chunk)))

            log_step(
                f"[features] running parallel Rscript workers={feature_workers} "
                f"chunks={n_chunks} chunk_size={feature_chunk_size} rows={len(unique)}"
            )
            active: list[tuple[int, subprocess.Popen, Path, int]] = []
            completed = 0
            next_job = 0
            while next_job < len(jobs) or active:
                while next_job < len(jobs) and len(active) < feature_workers:
                    chunk_idx, chunk_in, chunk_out, n_rows = jobs[next_job]
                    cmd = ["Rscript", str(rscript_path), str(chunk_in), str(chunk_out), peptide_col]
                    log_step(f"[features] start chunk={chunk_idx + 1}/{n_chunks} rows={n_rows}")
                    proc = subprocess.Popen(cmd)
                    active.append((chunk_idx, proc, chunk_out, n_rows))
                    next_job += 1

                still_active = []
                for chunk_idx, proc, chunk_out, n_rows in active:
                    rc = proc.poll()
                    if rc is None:
                        still_active.append((chunk_idx, proc, chunk_out, n_rows))
                        continue
                    if rc != 0:
                        for _, other_proc, _, _ in still_active:
                            other_proc.terminate()
                        raise subprocess.CalledProcessError(rc, proc.args)
                    if not chunk_out.exists():
                        raise FileNotFoundError(f"Missing peptide feature chunk output: {chunk_out}")
                    completed += 1
                    log_step(f"[features] done chunk={chunk_idx + 1}/{n_chunks} completed={completed}/{n_chunks}")
                active = still_active
                if active:
                    time.sleep(1)

            header = True
            with out_csv.open("w", encoding="utf-8", newline="") as out_handle:
                for chunk_idx, _, chunk_out, _ in jobs:
                    with chunk_out.open("r", encoding="utf-8") as in_handle:
                        for line_no, line in enumerate(in_handle):
                            if line_no == 0:
                                if header:
                                    out_handle.write(line)
                                    header = False
                                continue
                            out_handle.write(line)
        log_step(f"[features] done: rows={len(unique)} elapsed={time.time() - t0:.1f}s")
    else:
        log_step(f"[features] cache hit: {out_csv.name} rows={len(unique)}")
    feat = pd.read_csv(out_csv)
    return df.merge(feat[[peptide_col] + FEATURE_COLUMNS], on=peptide_col, how="left")


def add_zero_peptide_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add an explicit zero x2 block for sequence-only Stage 1 encoding."""
    out = df.copy()
    out["aa_composition"] = ZERO_AA_COMPOSITION
    for column in FEATURE_COLUMNS[1:]:
        out[column] = 0.0
    return out


def aaindex_vocab(seq: str, vocab_map: Dict[str, int]) -> np.ndarray:
    encoded = np.empty([len(seq)], dtype=np.float32)
    for i, aa in enumerate(seq):
        query = "-" if aa.upper() == "X" else aa.upper()
        encoded[i] = vocab_map.get(query, 0)
    return encoded


def extract_feature_vector(row: pd.Series) -> List[float]:
    aa_comp = [float(value.strip()) for value in str(row["aa_composition"]).split(",")]
    other = [float(row[col]) for col in FEATURE_COLUMNS[1:]]
    features = aa_comp + other
    if len(features) != X2_DIM:
        raise ValueError(f"Expected {X2_DIM} peptide features, got {len(features)}")
    return features


def resolve_padding_length(
    requested_padding_length: int,
    hla_source: str,
    combined_df: pd.DataFrame,
    hla2seq: Dict[str, str],
) -> int:
    if requested_padding_length > 0:
        return int(requested_padding_length)
    if hla_source == "fasta":
        return PADDING_LENGTH
    max_peptide_len = int(combined_df["peptide"].astype(str).str.len().max())
    max_hla_len = max(len(seq) for seq in hla2seq.values())
    return X2_DIM + max_peptide_len + SEPARATOR_LENGTH + max_hla_len


def _encode_dataframe_chunk(args):
    chunk_df, hla2prot, vocab_map, padding_length = args
    encoded_rows: List[np.ndarray] = []
    labels: List[int] = []
    kept_indices: List[int] = []
    peptide_lengths: List[int] = []
    missing_hla = 0

    for idx, row in chunk_df.iterrows():
        hla_paratope = hla2prot.get(str(row["_norm_hla"]))
        if hla_paratope is None:
            missing_hla += 1
            continue
        feature_array = np.asarray(extract_feature_vector(row), dtype=np.float32)
        peptide_encode = aaindex_vocab(str(row["peptide"]), vocab_map)
        hla_encode = aaindex_vocab(hla_paratope, vocab_map)
        zero_array = np.zeros(SEPARATOR_LENGTH, dtype=np.float32)
        merged = np.concatenate((feature_array, peptide_encode, zero_array, hla_encode), axis=0)
        if merged.shape[0] > padding_length:
            merged = merged[:padding_length]
        pad_len = padding_length - merged.shape[0]
        encoded_rows.append(np.pad(merged, (0, pad_len), "constant").astype(np.float32))
        labels.append(int(row["label"]))
        kept_indices.append(int(idx))
        peptide_lengths.append(len(str(row["peptide"])))

    if encoded_rows:
        encoded = np.stack(encoded_rows, axis=0)
    else:
        encoded = np.empty((0, padding_length), dtype=np.float32)
    return encoded, labels, peptide_lengths, kept_indices, missing_hla


def _split_dataframe_rows(df: pd.DataFrame, chunk_size: int) -> List[pd.DataFrame]:
    if len(df) == 0:
        return []
    chunk_size = max(1, int(chunk_size))
    return [df.iloc[start : start + chunk_size].copy() for start in range(0, len(df), chunk_size)]


def encode_dataset(
    df: pd.DataFrame,
    hla2prot: Dict[str, str],
    vocab: Vocab,
    padding_length: int,
    encode_workers: int = 1,
    encode_chunk_size: int = 100000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
    reset_df = df.reset_index(drop=True)
    chunks = _split_dataframe_rows(reset_df, encode_chunk_size)
    log_step(
        f"[encode] rows={len(reset_df)} chunks={len(chunks)} "
        f"workers={encode_workers} chunk_size={encode_chunk_size}"
    )
    worker_args = [(chunk, hla2prot, vocab.token_to_idx, padding_length) for chunk in chunks]

    encoded_parts: List[np.ndarray] = []
    labels: List[int] = []
    peptide_lengths: List[int] = []
    kept_indices: List[int] = []
    missing_hla = 0
    if encode_workers <= 1 or len(chunks) <= 1:
        iterator = tqdm(worker_args, total=len(worker_args), desc="Encoding peptide-HLA chunks", leave=False)
        for args in iterator:
            encoded, chunk_labels, chunk_lengths, chunk_indices, chunk_missing = _encode_dataframe_chunk(args)
            if len(encoded):
                encoded_parts.append(encoded)
            labels.extend(chunk_labels)
            peptide_lengths.extend(chunk_lengths)
            kept_indices.extend(chunk_indices)
            missing_hla += chunk_missing
    else:
        max_workers = max(1, int(encode_workers))
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_encode_dataframe_chunk, args) for args in worker_args]
            for future in tqdm(
                futures,
                total=len(futures),
                desc=f"Encoding peptide-HLA chunks ({max_workers} workers)",
                leave=False,
            ):
                encoded, chunk_labels, chunk_lengths, chunk_indices, chunk_missing = future.result()
                if len(encoded):
                    encoded_parts.append(encoded)
                labels.extend(chunk_labels)
                peptide_lengths.extend(chunk_lengths)
                kept_indices.extend(chunk_indices)
                missing_hla += chunk_missing

    if missing_hla:
        log_step(f"[encode] skipped rows with missing HLA sequence: {missing_hla}")
    if not encoded_parts:
        raise RuntimeError("No rows could be encoded. Check HLA fasta coverage.")
    encoded = torch.from_numpy(np.concatenate(encoded_parts, axis=0))
    y = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(peptide_lengths, dtype=torch.long)
    kept = df.reset_index(drop=True).iloc[kept_indices].reset_index(drop=True)
    kept.attrs["_kept_indices"] = kept_indices
    return encoded, y, lengths, kept


def encode_dataset_cached(
    df: pd.DataFrame,
    hla2prot: Dict[str, str],
    vocab: Vocab,
    padding_length: int,
    cache_dir: Path,
    split_name: str,
    hla_source: str,
    hla_digest: str,
    enabled: bool = True,
    encode_workers: int = 1,
    encode_chunk_size: int = 100000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, pd.DataFrame]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    encoded_columns = ["peptide", "_norm_hla", "label"] + FEATURE_COLUMNS
    data_digest = sha_for_dataframe_columns(df, encoded_columns)
    cache_key_payload = {
        "version": ENCODING_CACHE_VERSION,
        "split": split_name,
        "data_digest": data_digest,
        "hla_source": hla_source,
        "hla_digest": hla_digest,
        "padding_length": int(padding_length),
        "vocab": AMINO,
        "x2_dim": X2_DIM,
        "separator_length": SEPARATOR_LENGTH,
    }
    key_hash = hashlib.sha256(
        json.dumps(cache_key_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    cache_path = cache_dir / f"encoded_{split_name}_{key_hash}.pt"

    reset_df = df.reset_index(drop=True)
    if enabled and cache_path.exists():
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(cache_path, map_location="cpu")
        metadata = payload.get("metadata", {})
        if metadata == cache_key_payload:
            kept_indices = payload["kept_indices"]
            kept = reset_df.iloc[kept_indices].reset_index(drop=True)
            log_step(
                f"[encode] cache hit split={split_name} file={cache_path.name} rows={len(kept_indices)}"
            )
            return payload["x"], payload["y"], payload["peptide_lengths"], kept
        log_step(f"[encode] cache metadata mismatch, rebuilding split={split_name}: {cache_path.name}")

    x, y, peptide_lengths, kept = encode_dataset(
        reset_df,
        hla2prot,
        vocab,
        padding_length,
        encode_workers=encode_workers,
        encode_chunk_size=encode_chunk_size,
    )
    if enabled:
        kept_indices = [int(i) for i in kept.attrs.get("_kept_indices", kept.index.tolist())]
        torch.save(
            {
                "x": x.cpu(),
                "y": y.cpu(),
                "peptide_lengths": peptide_lengths.cpu(),
                "kept_indices": kept_indices,
                "metadata": cache_key_payload,
            },
            cache_path,
        )
        log_step(f"[encode] cache write split={split_name} file={cache_path.name} rows={len(kept_indices)}")
    return x, y, peptide_lengths, kept


def apply_physchem_mode(features: torch.Tensor, mode: str) -> torch.Tensor:
    """Return encoded features with the 25-dimensional x2 block configured for an ablation."""
    if mode == "full":
        return features
    if mode != "zero":
        raise ValueError(f"Unsupported physicochemical feature mode: {mode}")
    ablated = features.clone()
    ablated[:, :X2_DIM] = 0
    if torch.count_nonzero(ablated[:, :X2_DIM]).item() != 0:
        raise RuntimeError("Physicochemical feature ablation failed to zero the x2 block")
    return ablated


def fit_physchem_scaler(features: torch.Tensor) -> Dict[str, List[float]]:
    x2 = features[:, :X2_DIM].float()
    if not torch.isfinite(x2).all():
        raise ValueError("Cannot fit physicochemical scaler with NaN or infinite values")
    mean = x2.mean(dim=0)
    scale = x2.std(dim=0, unbiased=False)
    zero_variance = scale < 1e-8
    scale = scale.clone()
    scale[zero_variance] = 1.0
    return {
        "mean": mean.cpu().tolist(),
        "scale": scale.cpu().tolist(),
        "zero_variance_dimensions": torch.where(zero_variance)[0].cpu().tolist(),
        "n_fit": int(x2.shape[0]),
    }


def load_physchem_scaler(path: Path) -> Dict[str, List[float]]:
    scaler = json.loads(path.read_text(encoding="utf-8"))
    for key in ("mean", "scale"):
        if key not in scaler or len(scaler[key]) != X2_DIM:
            raise ValueError(f"Invalid physicochemical scaler {path}: {key} must have {X2_DIM} values")
    return scaler


def load_checkpoint_physchem_scaler(path: Path, device: torch.device) -> Dict[str, List[float]]:
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or "metadata" not in payload:
        raise RuntimeError(f"Checkpoint does not contain bound metadata: {path}")
    scaler = payload["metadata"].get("physchem_scaler")
    if scaler is None:
        raise RuntimeError(f"Checkpoint does not contain a bound physicochemical scaler: {path}")
    for key in ("mean", "scale"):
        if key not in scaler or len(scaler[key]) != X2_DIM:
            raise ValueError(f"Invalid checkpoint-bound scaler {path}: {key} must have {X2_DIM} values")
    return dict(scaler)


def transform_physchem_features(
    features: torch.Tensor,
    scaler: Dict[str, List[float]],
    clip: float,
) -> torch.Tensor:
    transformed = features.clone()
    mean = torch.tensor(scaler["mean"], dtype=transformed.dtype, device=transformed.device)
    scale = torch.tensor(scaler["scale"], dtype=transformed.dtype, device=transformed.device)
    x2 = (transformed[:, :X2_DIM] - mean) / scale
    if clip > 0:
        x2 = x2.clamp(min=-clip, max=clip)
    if not torch.isfinite(x2).all():
        raise ValueError("Physicochemical standardization produced NaN or infinite values")
    transformed[:, :X2_DIM] = x2
    return transformed


def initialize_model(
    model_cls,
    after_pca_path: Path,
    device: torch.device,
    *,
    architecture: str,
    source_count: int = 0,
    source_embedding_dim: int = 8,
    physchem_fusion: str = "legacy_concat",
    physchem_adapter_dim: int = 64,
    physchem_dropout: float = 0.2,
    refiner_embedding_dim: int = 64,
    refiner_composition_dim: int = 18,
    refiner_composition_hidden_dim: int = 48,
    refiner_scalar_hidden_dim: int = 32,
    refiner_fusion_hidden_dim: int = 128,
    refiner_gate_hidden_dim: int = 64,
    refiner_dropout: float = 0.1,
    refiner_layer_scale_init: float = 0.1,
    late_physchem_dim: int = 32,
    late_group_hidden_dim: int = 16,
    late_gate_hidden_dim: int = 64,
    late_alpha_init: float = 0.1,
    late_alpha_max: float = 0.5,
    film_physchem_dim: int = 32,
    film_group_hidden_dim: int = 16,
    film_scale: float = 0.1,
) -> nn.Module:
    after_pca = np.loadtxt(after_pca_path)
    zero_row = np.zeros((1, after_pca.shape[1]), dtype=np.float32)
    embeds = torch.tensor(np.vstack([zero_row, after_pca]), dtype=torch.float32)
    vocab = Vocab(list(AMINO), min_freq=1)
    common_kwargs = {
        "vocab_size": len(vocab),
        "embedding_dim": 12,
        "hidden_dim_x1": 256,
        "hidden_dim_x2": 16,
        "output_dim": 2,
        "n_layers": 2,
        "bidirectional": True,
        "dropout": 0.5,
        "pad_idx": vocab["<pad>"],
        "x2_dim": 25,
        "physchem_fusion": physchem_fusion,
        "physchem_adapter_dim": physchem_adapter_dim,
        "physchem_dropout": physchem_dropout,
        "refiner_embedding_dim": refiner_embedding_dim,
        "refiner_composition_dim": refiner_composition_dim,
        "refiner_composition_hidden_dim": refiner_composition_hidden_dim,
        "refiner_scalar_hidden_dim": refiner_scalar_hidden_dim,
        "refiner_fusion_hidden_dim": refiner_fusion_hidden_dim,
        "refiner_gate_hidden_dim": refiner_gate_hidden_dim,
        "refiner_dropout": refiner_dropout,
        "refiner_layer_scale_init": refiner_layer_scale_init,
        "late_physchem_dim": late_physchem_dim,
        "late_group_hidden_dim": late_group_hidden_dim,
        "late_gate_hidden_dim": late_gate_hidden_dim,
        "late_alpha_init": late_alpha_init,
        "late_alpha_max": late_alpha_max,
        "film_physchem_dim": film_physchem_dim,
        "film_group_hidden_dim": film_group_hidden_dim,
        "film_scale": film_scale,
    }
    if architecture == "source_aware":
        if source_count <= 0:
            raise ValueError("source-aware architecture requires source_count > 0")
        model = model_cls(
            **common_kwargs,
            source_count=source_count,
            source_embedding_dim=source_embedding_dim,
        )
    else:
        model = model_cls(
            **common_kwargs,
        )
    model.embedding.weight.data.copy_(embeds)
    model.embedding.weight.requires_grad = False
    return model.to(device)


def unpack_batch(batch, device: torch.device):
    if len(batch) == 5:
        features, labels, peptide_lengths, physchem_present, source_ids = batch
        source_ids = source_ids.to(device)
    elif len(batch) == 4:
        features, labels, peptide_lengths, physchem_present = batch
        source_ids = None
    elif len(batch) in {2, 3}:
        # Backward-compatible loader contract for legacy checkpoints only.
        features, labels = batch[:2]
        source_ids = batch[2].to(device) if len(batch) == 3 else None
        peptide_lengths = None
        physchem_present = None
    else:
        raise ValueError(f"Unsupported batch structure with {len(batch)} tensors")
    features = features.to(device)
    labels = labels.to(device)
    if peptide_lengths is not None:
        peptide_lengths = peptide_lengths.to(device)
        physchem_present = physchem_present.to(device)
    return features, labels, peptide_lengths, physchem_present, source_ids


def model_logits(
    model: nn.Module,
    features: torch.Tensor,
    peptide_lengths: torch.Tensor | None = None,
    physchem_present: torch.Tensor | None = None,
    source_ids: torch.Tensor | None = None,
    return_diagnostics: bool = False,
    return_components: bool = False,
):
    x1 = features[:, X2_DIM:].long()
    x2 = features[:, 0:X2_DIM].float()
    kwargs = {
        "peptide_lengths": peptide_lengths,
        "physchem_present": physchem_present,
        "return_diagnostics": return_diagnostics,
    }
    if return_components:
        kwargs["return_components"] = True
    if source_ids is None:
        return model(x1, x2, **kwargs)
    return model(x1, x2, source_ids, **kwargs)


def select_prediction_logits(output, prediction_head: str = "fused") -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        if prediction_head not in {"fused", "sequence"}:
            raise ValueError(f"Model does not expose prediction head: {prediction_head}")
        return output
    if not isinstance(output, dict):
        raise TypeError(f"Unsupported model output type: {type(output)!r}")
    key = {
        "fused": "fused_logits",
        "sequence": "sequence_logits",
        "physchem": "physchem_logits",
    }[prediction_head]
    return output[key]


def _migrate_legacy_phla_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> List[str]:
    """Load a legacy late-concatenation checkpoint into the residual pHLA model."""
    target = model.state_dict()
    migrated = OrderedDict((key, value.clone()) for key, value in target.items())
    loaded: List[str] = []
    for key, target_value in target.items():
        source_value = state.get(key)
        if source_value is None and key.startswith("encoder."):
            source_value = state.get(key.removeprefix("encoder."))
        if source_value is not None and source_value.shape == target_value.shape:
            migrated[key] = source_value
            loaded.append(key)

    source_fc = state.get("fc.weight")
    target_fc = target.get("fc.weight")
    if (
        source_fc is not None
        and target_fc is not None
        and source_fc.ndim == 2
        and target_fc.ndim == 2
        and source_fc.shape[0] == target_fc.shape[0]
        and source_fc.shape[1] > target_fc.shape[1]
    ):
        migrated["fc.weight"] = source_fc[:, : target_fc.shape[1]].clone()
        if "fc.weight" not in loaded:
            loaded.append("fc.weight")

    required_prefixes = ("encoder.embedding.", "encoder.lstm.")
    if not all(any(key.startswith(prefix) for key in loaded) for prefix in required_prefixes):
        raise RuntimeError("Legacy checkpoint migration did not recover embedding and LSTM weights")
    if "fc.weight" not in loaded or "fc.bias" not in loaded:
        raise RuntimeError("Legacy checkpoint migration did not recover the sequence classifier")
    model.load_state_dict(migrated, strict=True)
    return loaded


def _load_encoder_only_weights(
    model: nn.Module,
    state: Dict[str, torch.Tensor],
) -> List[str]:
    normalized = OrderedDict()
    for key, value in state.items():
        clean = key.replace("module.", "", 1)
        if clean.startswith("embedding.") or clean.startswith("lstm."):
            clean = f"encoder.{clean}"
        normalized[clean] = value

    target = model.state_dict()
    required = [
        key
        for key in target
        if key.startswith("encoder.embedding.") or key.startswith("encoder.lstm.")
    ]
    loaded: List[str] = []
    for key in required:
        source_value = normalized.get(key)
        if source_value is not None and source_value.shape == target[key].shape:
            target[key] = source_value
            loaded.append(key)
    missing = sorted(set(required) - set(loaded))
    if missing:
        raise RuntimeError(
            "Could not recover the complete sequence encoder from the checkpoint; "
            f"missing={missing}"
        )
    model.load_state_dict(target, strict=True)
    return loaded


def _load_encoder_and_source_weights(
    model: nn.Module,
    state: Dict[str, torch.Tensor],
) -> List[str]:
    """Load the shared sequence encoder and source embedding, but not task heads."""
    if not hasattr(model, "source_embedding"):
        raise RuntimeError("encoder_source initialization requires a source-aware model")

    normalized = OrderedDict()
    for key, value in state.items():
        clean = key.replace("module.", "", 1)
        if clean.startswith("embedding.") or clean.startswith("lstm."):
            clean = f"encoder.{clean}"
        normalized[clean] = value

    target = model.state_dict()
    required = [
        key
        for key in target
        if key.startswith("encoder.embedding.")
        or key.startswith("encoder.embedding_projection.")
        or key.startswith("encoder.lstm.")
        or key.startswith("source_embedding.")
    ]
    loaded: List[str] = []
    for key in required:
        source_value = normalized.get(key)
        if source_value is not None and source_value.shape == target[key].shape:
            target[key] = source_value
            loaded.append(key)
    missing = sorted(set(required) - set(loaded))
    if missing:
        raise RuntimeError(
            "Could not recover the complete sequence encoder and source embedding from "
            f"the checkpoint; missing={missing}"
        )
    model.load_state_dict(target, strict=True)
    return loaded


def load_initial_weights(
    model: nn.Module,
    model_path: Path,
    device: torch.device,
    scope: str = "all",
) -> None:
    payload = torch.load(model_path, map_location=device)
    metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint format: {model_path}")
    if scope == "encoder":
        loaded = _load_encoder_only_weights(model, state)
        log_step(
            f"[init] loaded sequence encoder only: loaded={len(loaded)} "
            "temporary_head_and_physchem_branch_skipped=True"
        )
        return
    if scope == "encoder_source":
        loaded = _load_encoder_and_source_weights(model, state)
        log_step(
            f"[init] loaded sequence encoder and source embedding: loaded={len(loaded)} "
            "temporary_head_and_physchem_branch_skipped=True"
        )
        return
    if scope != "all":
        raise ValueError(f"Unsupported checkpoint loading scope: {scope}")
    if getattr(model, "physchem_fusion", None) == "embedding_refine":
        source_fusion = metadata.get("model_config", {}).get("physchem_fusion")
        if source_fusion != "embedding_refine":
            raise RuntimeError(
                "embedding_refine models may only initialize from an embedding_refine checkpoint; "
                "legacy Stage 1 weights are intentionally not reusable"
            )
    target_fusion = getattr(model, "physchem_fusion", None)
    if target_fusion in {"gated_late", "film_aux"}:
        source_fusion = metadata.get("model_config", {}).get("physchem_fusion")
        source_mode = metadata.get("physchem_mode")
        if source_fusion == "embedding_refine" and source_mode == "zero":
            state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
            normalized = OrderedDict((key.replace("module.", "", 1), value) for key, value in state.items())
            target = model.state_dict()
            prefixes = (
                "encoder.embedding.",
                "encoder.embedding_projection.",
                "encoder.lstm.",
            )
            required = [key for key in target if key.startswith(prefixes)]
            loaded = []
            for key in required:
                source_value = normalized.get(key)
                if source_value is not None and source_value.shape == target[key].shape:
                    target[key] = source_value
                    loaded.append(key)
            missing = sorted(set(required) - set(loaded))
            if missing:
                raise RuntimeError(
                    "Could not recover the complete sequence encoder from the embedding_refine "
                    f"Zero checkpoint; missing={missing}"
                )
            model.load_state_dict(target, strict=True)
            log_step(
                f"[init] loaded sequence encoder from embedding_refine Zero checkpoint: "
                f"loaded={len(loaded)} {target_fusion}_new_parameters=True"
            )
            return
        if source_fusion != target_fusion:
            raise RuntimeError(
                f"{target_fusion} models may initialize from an embedding_refine Zero Stage 1 "
                f"checkpoint or another {target_fusion} checkpoint"
            )
    candidates = [state]

    stripped = OrderedDict()
    for key, value in state.items():
        stripped[key.replace("module.", "", 1)] = value
    candidates.append(stripped)

    prefixed = OrderedDict()
    for key, value in state.items():
        prefixed[key if key.startswith("module.") else f"module.{key}"] = value
    candidates.append(prefixed)
    remapped = OrderedDict()
    for key, value in state.items():
        clean = key.replace("module.", "", 1)
        if clean.startswith("embedding.") or clean.startswith("lstm."):
            clean = f"encoder.{clean}"
        remapped[clean] = value
    candidates.append(remapped)

    last_error = None
    for candidate in candidates:
        try:
            model.load_state_dict(candidate, strict=True)
            return
        except RuntimeError as exc:
            last_error = exc
    if getattr(model, "physchem_fusion", None) == "residual":
        normalized = OrderedDict((key.replace("module.", "", 1), value) for key, value in state.items())
        loaded = _migrate_legacy_phla_state(model, normalized)
        log_step(
            f"[init] migrated legacy late-concatenation checkpoint: loaded={len(loaded)} "
            f"new_adapter_zero_initialized=True"
        )
        return
    raise RuntimeError(f"Failed to load initial weights from {model_path}. Last error: {last_error}")


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).sum().item())


def evaluate(
    model: nn.Module,
    loader: data.DataLoader,
    device: torch.device,
    prediction_head: str = "fused",
) -> Dict[str, float]:
    model.eval()
    labels_all: List[int] = []
    probs_all: List[float] = []
    pred_all: List[int] = []
    correct = 0.0
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            features, labels, peptide_lengths, physchem_present, source_ids = unpack_batch(batch, device)
            output = model_logits(
                model,
                features,
                peptide_lengths,
                physchem_present,
                source_ids,
                return_components=prediction_head != "fused",
            )
            logits = select_prediction_logits(output, prediction_head)
            probs = torch.softmax(logits, dim=1)[:, 1]
            correct += accuracy(logits, labels)
            total += labels.numel()
            labels_all.extend(labels.detach().cpu().numpy().astype(int).tolist())
            probs_all.extend(probs.detach().cpu().numpy().astype(float).tolist())
            pred_all.extend(logits.argmax(dim=1).detach().cpu().numpy().astype(int).tolist())
    metrics = {"accuracy": correct / max(total, 1)}
    if len(set(labels_all)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(labels_all, probs_all))
        metrics["average_precision"] = float(average_precision_score(labels_all, probs_all))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")
    return metrics


def predict(
    model: nn.Module,
    loader: data.DataLoader,
    device: torch.device,
    prediction_head: str = "fused",
) -> Tuple[List[int], List[float], List[int]]:
    model.eval()
    labels_all: List[int] = []
    probs_all: List[float] = []
    pred_all: List[int] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", leave=False):
            features, labels, peptide_lengths, physchem_present, source_ids = unpack_batch(batch, device)
            output = model_logits(
                model,
                features,
                peptide_lengths,
                physchem_present,
                source_ids,
                return_components=prediction_head != "fused",
            )
            logits = select_prediction_logits(output, prediction_head)
            probs = torch.softmax(logits, dim=1)[:, 1]
            labels_all.extend(labels.detach().cpu().numpy().astype(int).tolist())
            probs_all.extend(probs.detach().cpu().numpy().astype(float).tolist())
            pred_all.extend(logits.argmax(dim=1).detach().cpu().numpy().astype(int).tolist())
    return labels_all, probs_all, pred_all


def collect_refiner_diagnostics(
    model: nn.Module,
    loader: data.DataLoader,
    device: torch.device,
    max_batches: int = 32,
) -> Dict[str, float]:
    fusion = getattr(model, "physchem_fusion", None)
    if fusion not in {"embedding_refine", "gated_late", "film_aux"}:
        return {}
    model.eval()
    if fusion == "embedding_refine":
        keys = [
            "base_embedding_norm",
            "delta_norm",
            "delta_to_base_ratio",
            "gate_mean",
            "gate_std",
            "gate_min",
            "gate_max",
            "gamma_norm",
            "beta_norm",
            "layer_scale_norm",
        ]
    elif fusion == "gated_late":
        keys = [
            "base_representation_norm",
            "delta_norm",
            "delta_to_base_ratio",
            "physchem_embedding_norm",
            "gate_mean",
            "gate_std",
            "gate_min",
            "gate_max",
            "alpha",
        ]
    else:
        keys = [
            "base_representation_norm",
            "modulation_norm",
            "modulation_to_base_ratio",
            "physchem_embedding_norm",
            "gamma_abs_mean",
            "beta_abs_mean",
            "film_scale",
        ]
    values = {key: [] for key in keys}
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            features, _, peptide_lengths, physchem_present, source_ids = unpack_batch(batch, device)
            _, diagnostics = model_logits(
                model,
                features,
                peptide_lengths,
                physchem_present,
                source_ids,
                return_diagnostics=True,
            )
            for key in keys:
                values[key].append(float(diagnostics[key].detach().cpu()))
    summary = {key: float(np.mean(items)) for key, items in values.items() if items}
    summary["batches"] = min(len(loader), max_batches)
    return summary


def set_sequence_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return
    for name, parameter in encoder.named_parameters():
        if name == "embedding.weight":
            parameter.requires_grad = False
        else:
            parameter.requires_grad = trainable


def training_optimizer(model: nn.Module, head_lr: float, encoder_lr: float) -> torch.optim.Optimizer:
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return torch.optim.Adam(model.parameters(), lr=head_lr)
    encoder_parameters = list(encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    head_parameters = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
    return torch.optim.Adam(
        [
            {"params": encoder_parameters, "lr": encoder_lr},
            {"params": head_parameters, "lr": head_lr},
        ]
    )


def film_aux_multitask_loss(
    output: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    physchem_present: torch.Tensor,
    loss_fn: nn.Module,
    aux_physchem_loss_weight: float,
    aux_sequence_loss_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a controlled Full/Zero FiLM loss and its components.

    The denominator is deliberately identical in Full and Zero conditions.
    Otherwise disabling the physicochemical branch would increase the effective
    sequence learning rate and bias the ablation in favor of Zero.
    """
    fused_loss = loss_fn(output["fused_logits"], labels)
    sequence_loss = loss_fn(output["sequence_logits"], labels)
    physchem_loss = loss_fn(output["physchem_logits"], labels)
    present = physchem_present.to(dtype=physchem_loss.dtype)
    active_physchem_loss = physchem_loss * present
    normalizer = 1.0 + aux_sequence_loss_weight + aux_physchem_loss_weight
    loss = (
        fused_loss
        + aux_sequence_loss_weight * sequence_loss
        + aux_physchem_loss_weight * active_physchem_loss
    ) / normalizer
    return loss, fused_loss, sequence_loss, physchem_loss


def train_one_model(
    model: nn.Module,
    train_loader: data.DataLoader,
    val_loader: data.DataLoader | None,
    device: torch.device,
    epochs: int,
    lr: float,
    checkpoint_dir: Path,
    model_name: str,
    loss_reduction: str,
    class_weights: torch.Tensor | None = None,
    checkpoint_metadata: Dict | None = None,
    prediction_head: str = "fused",
    aux_physchem_loss_weight: float = 0.0,
    aux_sequence_loss_weight: float = 0.0,
    encoder_warmup_epochs: int = 0,
    encoder_lr: float | None = None,
) -> Tuple[nn.Module, List[Dict[str, float]], Dict[str, Path]]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    encoder_lr = lr if encoder_lr is None else encoder_lr
    trainer = training_optimizer(model, lr, encoder_lr)
    loss_fn = nn.CrossEntropyLoss(
        reduction="none",
        weight=class_weights.to(device) if class_weights is not None else None,
    )
    history: List[Dict[str, float]] = []
    best_values = {
        "val_roc_auc": -float("inf"),
        "val_average_precision": -float("inf"),
    }
    best_paths = {
        "val_roc_auc": checkpoint_dir / f"{model_name}_best_val_roc_auc.pth",
        "val_average_precision": checkpoint_dir / f"{model_name}_best_val_average_precision.pth",
    }
    uses_film_aux = getattr(model, "physchem_fusion", None) == "film_aux"
    if prediction_head != "fused" and not uses_film_aux:
        raise ValueError(f"prediction_head={prediction_head} requires physchem_fusion=film_aux")
    permanently_freeze_encoder = prediction_head == "physchem"
    for epoch in range(epochs):
        encoder_trainable = not permanently_freeze_encoder and epoch >= encoder_warmup_epochs
        set_sequence_encoder_trainable(model, encoder_trainable)
        if epoch == 0 or epoch == encoder_warmup_epochs:
            log_step(
                f"[{model_name}] encoder_trainable={encoder_trainable} "
                f"head_lr={lr} encoder_lr={encoder_lr}"
            )
        model.train()
        loss_sum = 0.0
        fused_loss_sum = 0.0
        sequence_loss_sum = 0.0
        physchem_loss_sum = 0.0
        correct = 0.0
        n = 0
        batch_iter = tqdm(
            train_loader,
            desc=f"{model_name} epoch {epoch + 1}/{epochs}",
            leave=False,
        )
        for batch in batch_iter:
            features, labels, peptide_lengths, physchem_present, source_ids = unpack_batch(batch, device)
            trainer.zero_grad()
            output = model_logits(
                model,
                features,
                peptide_lengths,
                physchem_present,
                source_ids,
                return_components=uses_film_aux,
            )
            logits = select_prediction_logits(output, prediction_head)
            if uses_film_aux:
                if prediction_head == "physchem":
                    fused_loss = loss_fn(output["fused_logits"], labels)
                    sequence_loss = loss_fn(output["sequence_logits"], labels)
                    physchem_loss = loss_fn(output["physchem_logits"], labels)
                    loss = physchem_loss
                elif prediction_head == "sequence":
                    fused_loss = loss_fn(output["fused_logits"], labels)
                    sequence_loss = loss_fn(output["sequence_logits"], labels)
                    physchem_loss = loss_fn(output["physchem_logits"], labels)
                    loss = sequence_loss
                else:
                    loss, fused_loss, sequence_loss, physchem_loss = film_aux_multitask_loss(
                        output,
                        labels,
                        physchem_present,
                        loss_fn,
                        aux_physchem_loss_weight,
                        aux_sequence_loss_weight,
                    )
                fused_loss_sum += float(fused_loss.sum().detach().cpu().item())
                sequence_loss_sum += float(sequence_loss.sum().detach().cpu().item())
                physchem_loss_sum += float(physchem_loss.sum().detach().cpu().item())
            else:
                loss = loss_fn(logits, labels)
            if loss_reduction == "mean":
                loss.mean().backward()
            else:
                loss.sum().backward()
            trainer.step()
            loss_sum += float(loss.sum().detach().cpu().item())
            correct += accuracy(logits, labels)
            n += labels.numel()
            batch_iter.set_postfix(
                loss=f"{loss_sum / max(n, 1):.4f}",
                acc=f"{correct / max(n, 1):.4f}",
            )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(n, 1),
            "train_accuracy": correct / max(n, 1),
        }
        if uses_film_aux:
            row.update(
                {
                    "train_fused_loss": fused_loss_sum / max(n, 1),
                    "train_sequence_loss": sequence_loss_sum / max(n, 1),
                    "train_physchem_loss": physchem_loss_sum / max(n, 1),
                }
            )
        if val_loader is not None:
            row.update(
                {
                    f"val_{k}": v
                    for k, v in evaluate(model, val_loader, device, prediction_head).items()
                }
            )
            for metric_name, best_path in best_paths.items():
                metric_value = float(row.get(metric_name, float("nan")))
                if np.isfinite(metric_value) and metric_value > best_values[metric_name]:
                    best_values[metric_name] = metric_value
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state": model.state_dict(),
                            "optimizer_state": trainer.state_dict(),
                            "history": history + [row],
                            "best_metric": metric_name,
                            "best_value": metric_value,
                            "metadata": checkpoint_metadata or {},
                        },
                        best_path,
                    )
        history.append(row)
        msg = (
            f"[{model_name}] epoch={epoch + 1}/{epochs} "
            f"loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f}"
        )
        if val_loader is not None:
            msg += (
                f" val_acc={row.get('val_accuracy', float('nan')):.4f}"
                f" val_auc={row.get('val_roc_auc', float('nan')):.4f}"
                f" val_ap={row.get('val_average_precision', float('nan')):.4f}"
            )
        log_step(msg)
        if (epoch + 1) % 100 == 0 or epoch == epochs - 1:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": trainer.state_dict(),
                    "history": history,
                    "metadata": checkpoint_metadata or {},
                },
                checkpoint_dir / f"{model_name}_epoch_{epoch}.pth",
            )
    return model, history, best_paths


def old_notebook_kfold_indices(labels: Sequence[int], k: int) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    labels_np = np.asarray(labels)
    class_indices = [np.where(labels_np == cls)[0].tolist() for cls in sorted(set(labels_np.tolist()))]
    fold_chunks: List[List[List[int]]] = []
    for idxs in class_indices:
        chunk_size = int(len(idxs) / k)
        chunks = []
        start = 0
        for fold in range(k):
            if fold != k - 1:
                chunks.append(idxs[start : start + chunk_size])
                start += chunk_size
            else:
                chunks.append(idxs[start:])
        fold_chunks.append(chunks)
    for fold in range(k):
        test_idx: List[int] = []
        train_idx: List[int] = []
        for chunks in fold_chunks:
            for i, chunk in enumerate(chunks):
                if i == fold:
                    test_idx.extend(chunk)
                else:
                    train_idx.extend(chunk)
        yield np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int)


def source_ids_from_metadata(
    kept_df: pd.DataFrame,
    source_col: str,
    source_to_idx: Dict[str, int],
) -> torch.Tensor:
    if source_col not in kept_df.columns:
        raise KeyError(f"Missing source column for source-aware model: {source_col}")
    values = kept_df[source_col].astype(str).tolist()
    missing = sorted(set(values) - set(source_to_idx))
    if missing:
        raise KeyError(f"Source mapping missing values from {source_col}: {missing}")
    return torch.tensor([source_to_idx[value] for value in values], dtype=torch.long)


def make_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    peptide_lengths: torch.Tensor,
    physchem_present: torch.Tensor,
    source_ids: torch.Tensor | None = None,
) -> data.DataLoader:
    if source_ids is None:
        ds = data.TensorDataset(
            features[indices], labels[indices], peptide_lengths[indices], physchem_present[indices]
        )
    else:
        ds = data.TensorDataset(
            features[indices], labels[indices], peptide_lengths[indices], physchem_present[indices], source_ids[indices]
        )
    return data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def make_training_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    kept_df: pd.DataFrame,
    batch_size: int,
    sampler_mode: str,
    sampler_source_col: str,
    outdir: Path,
    peptide_lengths: torch.Tensor,
    physchem_present: torch.Tensor,
    source_ids: torch.Tensor | None = None,
) -> data.DataLoader:
    if source_ids is None:
        ds = data.TensorDataset(features, labels, peptide_lengths, physchem_present)
    else:
        ds = data.TensorDataset(features, labels, peptide_lengths, physchem_present, source_ids)
    if sampler_mode == "shuffle":
        return data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    meta = kept_df.reset_index(drop=True).copy()
    if len(meta) != len(labels):
        raise ValueError(f"Sampler metadata rows ({len(meta)}) != labels ({len(labels)})")
    meta["_label"] = labels.detach().cpu().numpy().astype(int)
    if sampler_mode == "label_balanced":
        group_cols = ["_label"]
    elif sampler_mode == "source_label_balanced":
        if sampler_source_col not in meta.columns:
            raise KeyError(f"Missing sampler source column: {sampler_source_col}")
        group_cols = [sampler_source_col, "_label"]
    else:
        raise ValueError(f"Unsupported train sampler: {sampler_mode}")

    group_sizes = meta.groupby(group_cols, dropna=False).size().rename("n").reset_index()
    meta = meta.merge(group_sizes, on=group_cols, how="left")
    weights = (1.0 / meta["n"].astype(float)).to_numpy(dtype=np.float64)
    sampler = data.WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
    report_path = outdir / "metrics" / "train_sampler_groups.tsv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    group_sizes.to_csv(report_path, sep="\t", index=False)
    log_step(
        f"[sampler] mode={sampler_mode} groups={len(group_sizes)} "
        f"source_col={sampler_source_col} report={report_path}"
    )
    return data.DataLoader(ds, batch_size=batch_size, sampler=sampler, drop_last=False)


def compute_class_weights(labels: torch.Tensor, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    if mode != "balanced":
        raise ValueError(f"Unsupported class weight mode: {mode}")
    label_np = labels.detach().cpu().numpy().astype(int)
    counts = np.bincount(label_np, minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Cannot compute balanced class weights with empty class: {counts.tolist()}")
    weights = counts.sum() / (len(counts) * counts)
    out = torch.tensor(weights, dtype=torch.float32)
    log_step(f"[class_weight] mode=balanced counts={counts.astype(int).tolist()} weights={weights.tolist()}")
    return out


def write_metrics(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--antigen", required=True)
    parser.add_argument("--train-csv", "--train-table", dest="train_csv", required=True, type=Path)
    parser.add_argument("--val-csv", "--val-table", dest="val_csv", required=True, type=Path)
    parser.add_argument("--test-csv", "--test-table", dest="test_csv", default=None, type=Path)
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument("--model-file", default=DEFAULT_MODEL_FILE, type=Path)
    parser.add_argument(
        "--architecture",
        choices=["phla", "source_aware"],
        default="phla",
        help="Model architecture for architecture-ablation experiments.",
    )
    parser.add_argument(
        "--source-col",
        default="antigen_class",
        help="Column used as source label when --architecture source_aware.",
    )
    parser.add_argument(
        "--source-labels",
        nargs="*",
        default=None,
        help="Optional stable source label order for source-aware embeddings.",
    )
    parser.add_argument("--source-embedding-dim", default=8, type=int)
    parser.add_argument(
        "--hla-source",
        choices=["fasta", "netmhc-pseudoseq"],
        default="fasta",
        help="Use original full-length HLA fasta or local NetMHC pseudo-sequence CSV resources.",
    )
    parser.add_argument("--hla-fasta", default=DEFAULT_HLA_FASTA, type=Path)
    parser.add_argument(
        "--hla-pseudoseq-csv",
        action="append",
        default=None,
        type=Path,
        help="NetMHC pseudo-sequence CSV with allele,pseudo_sequence columns. Can be passed multiple times.",
    )
    parser.add_argument(
        "--padding-length",
        default=0,
        type=int,
        help="Sequence padding length. 0 keeps 419 for fasta and auto-sizes for NetMHC pseudo-sequences.",
    )
    parser.add_argument("--after-pca", default=DEFAULT_AFTER_PCA, type=Path)
    parser.add_argument("--feature-rscript", default=DEFAULT_FEATURE_RSCRIPT, type=Path)
    parser.add_argument("--feature-workers", default=1, type=int)
    parser.add_argument("--feature-chunk-size", default=250000, type=int)
    parser.add_argument("--encode-workers", default=1, type=int)
    parser.add_argument("--encode-chunk-size", default=100000, type=int)
    parser.add_argument(
        "--physchem-mode",
        choices=["full", "zero"],
        default="full",
        help="Keep the 25-dimensional peptide physicochemical block or replace it with zeros.",
    )
    parser.add_argument(
        "--physchem-fusion",
        choices=["sequence_only", "legacy_concat", "residual", "embedding_refine", "gated_late", "film_aux"],
        default="legacy_concat",
        help=(
            "Physicochemical fusion strategy; embedding_refine modifies peptide token embeddings, "
            "gated_late conditions the pooled pHLA representation, and film_aux adds "
            "auxiliary-supervised residual FiLM."
        ),
    )
    parser.add_argument(
        "--physchem-scaling",
        choices=["none", "standardize"],
        default="none",
        help="Optionally standardize each physicochemical dimension using training-only statistics.",
    )
    parser.add_argument(
        "--physchem-scaler-in",
        default=None,
        type=Path,
        help="Optional saved training-only scaler. If omitted, standardize mode fits on the current train split.",
    )
    parser.add_argument("--physchem-clip", default=5.0, type=float)
    parser.add_argument("--physchem-adapter-dim", default=64, type=int)
    parser.add_argument("--physchem-dropout", default=0.2, type=float)
    parser.add_argument("--refiner-embedding-dim", default=64, type=int)
    parser.add_argument("--refiner-composition-dim", default=18, type=int)
    parser.add_argument("--refiner-composition-hidden-dim", default=48, type=int)
    parser.add_argument("--refiner-scalar-hidden-dim", default=32, type=int)
    parser.add_argument("--refiner-fusion-hidden-dim", default=128, type=int)
    parser.add_argument("--refiner-gate-hidden-dim", default=64, type=int)
    parser.add_argument("--refiner-dropout", default=0.1, type=float)
    parser.add_argument("--refiner-layer-scale-init", default=0.1, type=float)
    parser.add_argument("--late-physchem-dim", default=32, type=int)
    parser.add_argument("--late-group-hidden-dim", default=16, type=int)
    parser.add_argument("--late-gate-hidden-dim", default=64, type=int)
    parser.add_argument("--late-alpha-init", default=0.1, type=float)
    parser.add_argument("--late-alpha-max", default=0.5, type=float)
    parser.add_argument("--film-physchem-dim", default=32, type=int)
    parser.add_argument("--film-group-hidden-dim", default=16, type=int)
    parser.add_argument("--film-scale", default=0.1, type=float)
    parser.add_argument(
        "--prediction-head",
        choices=["fused", "sequence", "physchem"],
        default="fused",
        help="Prediction head optimized and reported; sequence/physchem require film_aux.",
    )
    parser.add_argument("--aux-physchem-loss-weight", default=0.2, type=float)
    parser.add_argument("--aux-sequence-loss-weight", default=0.1, type=float)
    parser.add_argument("--encoder-warmup-epochs", default=0, type=int)
    parser.add_argument(
        "--encoder-lr",
        default=None,
        type=float,
        help="Optional sequence-encoder learning rate; defaults to --lr.",
    )
    parser.add_argument("--diagnostic-batches", default=32, type=int)
    parser.add_argument(
        "--cache-dir",
        default=None,
        type=Path,
        help="Optional shared feature/encoding cache directory. Defaults to OUTDIR/cache.",
    )
    parser.add_argument(
        "--no-encoded-cache",
        dest="encoded_cache",
        action="store_false",
        help="Disable cached encoded tensor reuse.",
    )
    parser.set_defaults(encoded_cache=True)
    parser.add_argument("--init-from", default=None, type=Path)
    parser.add_argument(
        "--init-scope",
        choices=["all", "encoder", "encoder_source"],
        default="all",
        help=(
            "Load a complete compatible checkpoint, only its sequence encoder, or the "
            "sequence encoder plus source embedding for source-aware Stage 2 training."
        ),
    )
    parser.add_argument(
        "--select-best-metric",
        choices=["last", "val_roc_auc", "val_average_precision"],
        default="last",
        help="Checkpoint used for final validation/test prediction after training.",
    )
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument(
        "--loss-reduction",
        choices=["sum", "mean"],
        default="sum",
        help="Use sum to reproduce the original notebooks; use mean for batch-size-stable tuning.",
    )
    parser.add_argument(
        "--class-weight-mode",
        choices=["none", "balanced"],
        default="none",
        help="Optional class weights for CrossEntropyLoss.",
    )
    parser.add_argument(
        "--train-sampler",
        choices=["shuffle", "label_balanced", "source_label_balanced"],
        default="shuffle",
        help="Sampling strategy for the final training loader.",
    )
    parser.add_argument(
        "--sampler-source-col",
        default="dataset_source",
        help="Metadata column used by --train-sampler source_label_balanced.",
    )
    parser.add_argument("--kfold", default=5, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-rows-per-label", default=0, type=int)
    parser.add_argument("--skip-kfold", action="store_true")
    args = parser.parse_args()

    if args.physchem_fusion in {"embedding_refine", "gated_late", "film_aux"} and args.physchem_scaling != "standardize":
        parser.error(f"--physchem-fusion {args.physchem_fusion} requires --physchem-scaling standardize")
    if args.physchem_fusion == "sequence_only" and args.physchem_mode != "zero":
        parser.error("--physchem-fusion sequence_only requires --physchem-mode zero")
    if args.physchem_fusion == "sequence_only" and args.physchem_scaling != "none":
        parser.error("--physchem-fusion sequence_only requires --physchem-scaling none")
    if args.physchem_fusion == "gated_late" and not 0 < args.late_alpha_init < args.late_alpha_max:
        parser.error("gated_late requires 0 < --late-alpha-init < --late-alpha-max")
    if args.physchem_fusion == "film_aux" and args.film_scale <= 0:
        parser.error("film_aux requires --film-scale > 0")
    if args.prediction_head != "fused" and args.physchem_fusion != "film_aux":
        parser.error("--prediction-head sequence/physchem requires --physchem-fusion film_aux")
    if args.prediction_head == "physchem" and args.physchem_mode != "full":
        parser.error("--prediction-head physchem requires --physchem-mode full")
    if args.aux_physchem_loss_weight < 0 or args.aux_sequence_loss_weight < 0:
        parser.error("auxiliary loss weights must be non-negative")
    if args.encoder_warmup_epochs < 0:
        parser.error("--encoder-warmup-epochs must be non-negative")

    t0 = time.time()
    set_seed(args.seed)
    log_step(
        f"[start] antigen={args.antigen} architecture={args.architecture} "
        f"epochs={args.epochs} batch_size={args.batch_size} physchem_mode={args.physchem_mode} "
        f"physchem_fusion={args.physchem_fusion} physchem_scaling={args.physchem_scaling}"
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    for sub in ["cache", "models", "metrics", "predictions"]:
        (args.outdir / sub).mkdir(exist_ok=True)
    run_cache_dir = args.cache_dir if args.cache_dir is not None else args.outdir / "cache"
    run_cache_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    config["resolved_cache_dir"] = str(run_cache_dir)
    (args.outdir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available() else args.device)
    log_step(f"[env] device={device} cuda_available={torch.cuda.is_available()}")
    ModelClass = load_model_class(args.model_file, args.architecture)
    vocab = Vocab(list(AMINO), min_freq=1)
    hla2prot = load_hla_sequences(args.hla_source, args.hla_fasta, args.hla_pseudoseq_csv)
    hla_digest = sha_for_hla_sequences(hla2prot)
    hla_lengths = sorted({len(seq) for seq in hla2prot.values()})
    log_step(
        f"[resources] hla_source={args.hla_source} HLA alleles={len(hla2prot)} "
        f"hla_lengths={hla_lengths[:10]} hla_digest={hla_digest} after_pca={args.after_pca}"
    )

    train_df = standardize_input(read_table_auto(args.train_csv), "train")
    val_df = standardize_input(read_table_auto(args.val_csv), "validation")
    test_df = standardize_input(read_table_auto(args.test_csv), "test") if args.test_csv else None
    if args.smoke_rows_per_label > 0:
        train_df = (
            train_df.groupby("label", group_keys=False)
            .head(args.smoke_rows_per_label)
            .reset_index(drop=True)
        )
        val_df = (
            val_df.groupby("label", group_keys=False)
            .head(min(args.smoke_rows_per_label, 10))
            .reset_index(drop=True)
        )
        if test_df is not None:
            test_df = (
                test_df.groupby("label", group_keys=False)
                .head(min(args.smoke_rows_per_label, 10))
                .reset_index(drop=True)
            )

    split_frames = [train_df.assign(_split="train"), val_df.assign(_split="validation")]
    if test_df is not None:
        split_frames.append(test_df.assign(_split="test"))
    combined = pd.concat(split_frames, ignore_index=True)
    if args.architecture == "source_aware":
        if args.source_col not in combined.columns:
            raise KeyError(f"--architecture source_aware requires source column: {args.source_col}")
        source_labels = args.source_labels or sorted(combined[args.source_col].astype(str).dropna().unique().tolist())
        if not source_labels:
            raise ValueError("No source labels available for source-aware model.")
        source_to_idx = {str(label): idx for idx, label in enumerate(source_labels)}
        missing_sources = sorted(set(combined[args.source_col].astype(str)) - set(source_to_idx))
        if missing_sources:
            raise ValueError(f"Input contains sources not listed in --source-labels: {missing_sources}")
        (args.outdir / "source_mapping.json").write_text(
            json.dumps(
                {
                    "source_col": args.source_col,
                    "source_to_idx": source_to_idx,
                    "source_embedding_dim": args.source_embedding_dim,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        log_step(f"[source_aware] source_col={args.source_col} mapping={source_to_idx}")
    else:
        source_to_idx = {}
    if args.physchem_fusion == "sequence_only":
        combined = add_zero_peptide_features(combined)
        log_step("[features] sequence_only: skipped physicochemical feature calculation")
    else:
        combined = ensure_peptide_features(
            combined,
            run_cache_dir,
            args.feature_rscript,
            feature_workers=args.feature_workers,
            feature_chunk_size=args.feature_chunk_size,
        )
    padding_length = resolve_padding_length(args.padding_length, args.hla_source, combined, hla2prot)
    log_step(f"[encode] padding_length={padding_length}")
    train_feat = combined[combined["_split"] == "train"].drop(columns=["_split"]).reset_index(drop=True)
    val_feat = combined[combined["_split"] == "validation"].drop(columns=["_split"]).reset_index(drop=True)
    test_feat = (
        combined[combined["_split"] == "test"].drop(columns=["_split"]).reset_index(drop=True)
        if test_df is not None
        else None
    )

    log_step("[encode] training set")
    train_x, train_y, train_lengths, train_kept = encode_dataset_cached(
        train_feat,
        hla2prot,
        vocab,
        padding_length,
        run_cache_dir,
        "train",
        args.hla_source,
        hla_digest,
        enabled=args.encoded_cache,
        encode_workers=args.encode_workers,
        encode_chunk_size=args.encode_chunk_size,
    )
    log_step("[encode] validation set")
    val_x, val_y, val_lengths, val_kept = encode_dataset_cached(
        val_feat,
        hla2prot,
        vocab,
        padding_length,
        run_cache_dir,
        "validation",
        args.hla_source,
        hla_digest,
        enabled=args.encoded_cache,
        encode_workers=args.encode_workers,
        encode_chunk_size=args.encode_chunk_size,
    )
    if test_feat is not None:
        log_step("[encode] test set")
        test_x, test_y, test_lengths, test_kept = encode_dataset_cached(
            test_feat,
            hla2prot,
            vocab,
            padding_length,
            run_cache_dir,
            "test",
            args.hla_source,
            hla_digest,
            enabled=args.encoded_cache,
            encode_workers=args.encode_workers,
            encode_chunk_size=args.encode_chunk_size,
        )
    else:
        test_x, test_y, test_lengths, test_kept = None, None, None, None
    scaler = None
    if args.physchem_scaling == "standardize":
        if args.physchem_scaler_in is not None:
            scaler = load_physchem_scaler(args.physchem_scaler_in)
            scaler_fit_source = str(args.physchem_scaler_in)
        elif args.init_from is not None and args.physchem_fusion == "embedding_refine":
            scaler = load_checkpoint_physchem_scaler(args.init_from, device)
            scaler_fit_source = f"checkpoint:{args.init_from}"
            checkpoint_clip = float(scaler.get("clip", args.physchem_clip))
            if not math.isclose(checkpoint_clip, float(args.physchem_clip), rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(
                    f"--physchem-clip={args.physchem_clip} does not match checkpoint-bound clip={checkpoint_clip}"
                )
        else:
            scaler = fit_physchem_scaler(train_x)
            scaler_fit_source = "train_split"
        scaler.update(
            {
                "clip": float(args.physchem_clip),
                "fit_source": scaler_fit_source,
            }
        )
        scaler_path = args.outdir / "physchem_scaler.json"
        scaler_path.write_text(json.dumps(scaler, indent=2), encoding="utf-8")
        train_x = transform_physchem_features(train_x, scaler, args.physchem_clip)
        val_x = transform_physchem_features(val_x, scaler, args.physchem_clip)
        if test_x is not None:
            test_x = transform_physchem_features(test_x, scaler, args.physchem_clip)
        log_step(
            f"[physchem] standardized n_fit={scaler['n_fit']} clip={args.physchem_clip} "
            f"zero_variance={scaler['zero_variance_dimensions']} saved={scaler_path}"
        )
    train_x = apply_physchem_mode(train_x, args.physchem_mode)
    val_x = apply_physchem_mode(val_x, args.physchem_mode)
    if test_x is not None:
        test_x = apply_physchem_mode(test_x, args.physchem_mode)
    presence_value = 1.0 if args.physchem_mode == "full" else 0.0
    train_presence = torch.full((len(train_y),), presence_value, dtype=torch.float32)
    val_presence = torch.full((len(val_y),), presence_value, dtype=torch.float32)
    test_presence = (
        torch.full((len(test_y),), presence_value, dtype=torch.float32) if test_y is not None else None
    )
    log_step(
        f"[physchem] mode={args.physchem_mode} x2_dim={X2_DIM} "
        f"train_nonzero={int(torch.count_nonzero(train_x[:, :X2_DIM]).item())}"
    )
    log_step(
        f"[data] train rows={len(train_df)} encoded={len(train_kept)} "
        f"validation rows={len(val_df)} encoded={len(val_kept)} device={device}"
    )
    if args.architecture == "source_aware":
        train_source_ids = source_ids_from_metadata(train_kept, args.source_col, source_to_idx)
        val_source_ids = source_ids_from_metadata(val_kept, args.source_col, source_to_idx)
        test_source_ids = (
            source_ids_from_metadata(test_kept, args.source_col, source_to_idx)
            if test_kept is not None
            else None
        )
    else:
        train_source_ids = None
        val_source_ids = None
        test_source_ids = None
    class_weights = compute_class_weights(train_y, args.class_weight_mode)
    model_config = {
        "architecture": args.architecture,
        "embedding_dim": 12,
        "hidden_dim_x1": 256,
        "hidden_dim_x2": 16,
        "output_dim": 2,
        "n_layers": 2,
        "bidirectional": True,
        "dropout": 0.5,
        "x2_dim": X2_DIM,
        "physchem_fusion": args.physchem_fusion,
        "physchem_adapter_dim": args.physchem_adapter_dim,
        "physchem_dropout": args.physchem_dropout,
        "refiner_embedding_dim": args.refiner_embedding_dim,
        "refiner_composition_dim": args.refiner_composition_dim,
        "refiner_composition_hidden_dim": args.refiner_composition_hidden_dim,
        "refiner_scalar_hidden_dim": args.refiner_scalar_hidden_dim,
        "refiner_fusion_hidden_dim": args.refiner_fusion_hidden_dim,
        "refiner_gate_hidden_dim": args.refiner_gate_hidden_dim,
        "refiner_dropout": args.refiner_dropout,
        "refiner_layer_scale_init": args.refiner_layer_scale_init,
        "late_physchem_dim": args.late_physchem_dim,
        "late_group_hidden_dim": args.late_group_hidden_dim,
        "late_gate_hidden_dim": args.late_gate_hidden_dim,
        "late_alpha_init": args.late_alpha_init,
        "late_alpha_max": args.late_alpha_max,
        "film_physchem_dim": args.film_physchem_dim,
        "film_group_hidden_dim": args.film_group_hidden_dim,
        "film_scale": args.film_scale,
    }
    if args.architecture == "source_aware":
        model_config.update(
            source_count=len(source_to_idx),
            source_embedding_dim=args.source_embedding_dim,
        )
    scaler_digest = (
        hashlib.sha256(json.dumps(scaler, sort_keys=True).encode("utf-8")).hexdigest()
        if scaler is not None
        else None
    )
    checkpoint_metadata = {
        "format_version": "mimicneoai_immunogenicity_checkpoint_v2",
        "model_config": model_config,
        "physchem_scaler": scaler,
        "physchem_scaler_sha256": scaler_digest,
        "physchem_mode": args.physchem_mode,
        "prediction_head": args.prediction_head,
        "aux_physchem_loss_weight": args.aux_physchem_loss_weight,
        "aux_sequence_loss_weight": args.aux_sequence_loss_weight,
        "encoder_warmup_epochs": args.encoder_warmup_epochs,
        "encoder_lr": args.encoder_lr,
        "encoding_cache_version": ENCODING_CACHE_VERSION,
        "hla_source": args.hla_source,
        "padding_length": padding_length,
        "source_mapping": source_to_idx,
    }
    config["model_config"] = model_config
    config["physchem_scaler_sha256"] = scaler_digest
    (args.outdir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    init_kwargs = {
        "architecture": args.architecture,
        "physchem_fusion": args.physchem_fusion,
        "physchem_adapter_dim": args.physchem_adapter_dim,
        "physchem_dropout": args.physchem_dropout,
        "refiner_embedding_dim": args.refiner_embedding_dim,
        "refiner_composition_dim": args.refiner_composition_dim,
        "refiner_composition_hidden_dim": args.refiner_composition_hidden_dim,
        "refiner_scalar_hidden_dim": args.refiner_scalar_hidden_dim,
        "refiner_fusion_hidden_dim": args.refiner_fusion_hidden_dim,
        "refiner_gate_hidden_dim": args.refiner_gate_hidden_dim,
        "refiner_dropout": args.refiner_dropout,
        "refiner_layer_scale_init": args.refiner_layer_scale_init,
        "late_physchem_dim": args.late_physchem_dim,
        "late_group_hidden_dim": args.late_group_hidden_dim,
        "late_gate_hidden_dim": args.late_gate_hidden_dim,
        "late_alpha_init": args.late_alpha_init,
        "late_alpha_max": args.late_alpha_max,
        "film_physchem_dim": args.film_physchem_dim,
        "film_group_hidden_dim": args.film_group_hidden_dim,
        "film_scale": args.film_scale,
    }
    if args.architecture == "source_aware":
        init_kwargs.update(
            source_count=len(source_to_idx),
            source_embedding_dim=args.source_embedding_dim,
        )

    all_fold_rows: List[Dict[str, float]] = []
    if not args.skip_kfold:
        for fold, (train_idx, test_idx) in enumerate(old_notebook_kfold_indices(train_y.numpy(), args.kfold)):
            log_step(f"[kfold] fold={fold} train={len(train_idx)} test={len(test_idx)}")
            model = initialize_model(
                ModelClass,
                args.after_pca,
                device,
                **init_kwargs,
            )
            if args.init_from is not None:
                load_initial_weights(model, args.init_from, device, scope=args.init_scope)
            train_loader = make_loader(
                train_x,
                train_y,
                train_idx,
                args.batch_size,
                shuffle=True,
                peptide_lengths=train_lengths,
                physchem_present=train_presence,
                source_ids=train_source_ids,
            )
            test_loader = make_loader(
                train_x,
                train_y,
                test_idx,
                args.batch_size,
                shuffle=False,
                peptide_lengths=train_lengths,
                physchem_present=train_presence,
                source_ids=train_source_ids,
            )
            model, history, _ = train_one_model(
                model,
                train_loader,
                test_loader,
                device,
                args.epochs,
                args.lr,
                args.outdir / "models" / f"fold_{fold}",
                f"{args.antigen}_fold_{fold}",
                args.loss_reduction,
                class_weights,
                checkpoint_metadata,
                prediction_head=args.prediction_head,
                aux_physchem_loss_weight=args.aux_physchem_loss_weight,
                aux_sequence_loss_weight=args.aux_sequence_loss_weight,
                encoder_warmup_epochs=args.encoder_warmup_epochs,
                encoder_lr=args.encoder_lr,
            )
            final_metrics = evaluate(model, test_loader, device, args.prediction_head)
            final_metrics.update({"fold": fold, "n_train": len(train_idx), "n_test": len(test_idx)})
            all_fold_rows.append(final_metrics)
            write_metrics(args.outdir / "metrics" / f"fold_{fold}_history.tsv", history)
        write_metrics(args.outdir / "metrics" / "kfold_metrics.tsv", all_fold_rows)

    final_model = initialize_model(
        ModelClass,
        args.after_pca,
        device,
        **init_kwargs,
    )
    if args.init_from is not None:
        log_step(f"[init] loading weights from {args.init_from}")
        load_initial_weights(final_model, args.init_from, device, scope=args.init_scope)
    final_train_loader = make_training_loader(
        train_x,
        train_y,
        train_kept,
        args.batch_size,
        args.train_sampler,
        args.sampler_source_col,
        args.outdir,
        train_lengths,
        train_presence,
        source_ids=train_source_ids,
    )
    if val_source_ids is None:
        val_ds = data.TensorDataset(val_x, val_y, val_lengths, val_presence)
    else:
        val_ds = data.TensorDataset(val_x, val_y, val_lengths, val_presence, val_source_ids)
    val_loader = data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    final_model, final_history, final_best_paths = train_one_model(
        final_model,
        final_train_loader,
        val_loader,
        device,
        args.epochs,
        args.lr,
        args.outdir / "models" / "final_checkpoints",
        f"{args.antigen}_final",
        args.loss_reduction,
        class_weights,
        checkpoint_metadata,
        prediction_head=args.prediction_head,
        aux_physchem_loss_weight=args.aux_physchem_loss_weight,
        aux_sequence_loss_weight=args.aux_sequence_loss_weight,
        encoder_warmup_epochs=args.encoder_warmup_epochs,
        encoder_lr=args.encoder_lr,
    )
    selected_checkpoint = None
    if args.select_best_metric != "last":
        selected_checkpoint = final_best_paths[args.select_best_metric]
        log_step(f"[select] loading best checkpoint by {args.select_best_metric}: {selected_checkpoint}")
        load_initial_weights(final_model, selected_checkpoint, device, scope="all")
    torch.save(
        {"model_state": final_model.state_dict(), "metadata": checkpoint_metadata},
        args.outdir / "models" / f"MimicNeoAI_{args.antigen}_final.pth",
    )
    write_metrics(args.outdir / "metrics" / "final_history.tsv", final_history)

    labels, probs, preds = predict(final_model, val_loader, device, args.prediction_head)
    pred_df = val_kept.copy()
    pred_df["true_label"] = labels
    pred_df["pred_label"] = preds
    pred_df["immunogenicity_score"] = probs
    pred_df.to_csv(args.outdir / "predictions" / "validation_predictions.tsv", sep="\t", index=False)
    final_metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "roc_auc": float(roc_auc_score(labels, probs)) if len(set(labels)) == 2 else float("nan"),
        "average_precision": float(average_precision_score(labels, probs)) if len(set(labels)) == 2 else float("nan"),
        "n_validation": len(labels),
        "elapsed_seconds": time.time() - t0,
        "selected_checkpoint": str(selected_checkpoint) if selected_checkpoint is not None else "last",
    }
    write_metrics(args.outdir / "metrics" / "validation_metrics.tsv", [final_metrics])
    diagnostics = collect_refiner_diagnostics(
        final_model,
        val_loader,
        device,
        max_batches=args.diagnostic_batches,
    )
    if diagnostics:
        diagnostics.update(
            physchem_mode=args.physchem_mode,
            split="validation",
        )
        diagnostics_name = {
            "embedding_refine": "physchem_refiner_diagnostics.json",
            "gated_late": "physchem_gated_late_diagnostics.json",
            "film_aux": "physchem_film_aux_diagnostics.json",
        }.get(args.physchem_fusion, "physchem_diagnostics.json")
        diagnostics_path = args.outdir / "metrics" / diagnostics_name
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        log_step("[physchem_diagnostics] " + json.dumps(diagnostics, sort_keys=True))
    if test_x is not None and test_y is not None and test_kept is not None:
        if test_source_ids is None:
            test_ds = data.TensorDataset(test_x, test_y, test_lengths, test_presence)
        else:
            test_ds = data.TensorDataset(test_x, test_y, test_lengths, test_presence, test_source_ids)
        test_loader = data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
        test_labels, test_probs, test_preds = predict(
            final_model,
            test_loader,
            device,
            args.prediction_head,
        )
        test_pred_df = test_kept.copy()
        test_pred_df["true_label"] = test_labels
        test_pred_df["pred_label"] = test_preds
        test_pred_df["immunogenicity_score"] = test_probs
        test_pred_df.to_csv(args.outdir / "predictions" / "test_predictions.tsv", sep="\t", index=False)
        test_metrics = {
            "accuracy": float(accuracy_score(test_labels, test_preds)),
            "roc_auc": float(roc_auc_score(test_labels, test_probs)) if len(set(test_labels)) == 2 else float("nan"),
            "average_precision": float(average_precision_score(test_labels, test_probs)) if len(set(test_labels)) == 2 else float("nan"),
            "n_test": len(test_labels),
            "elapsed_seconds": time.time() - t0,
            "selected_checkpoint": str(selected_checkpoint) if selected_checkpoint is not None else "last",
        }
        write_metrics(args.outdir / "metrics" / "test_metrics.tsv", [test_metrics])
        log_step("[test] " + json.dumps(test_metrics, indent=2))
    log_step("[done] " + json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()
